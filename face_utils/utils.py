from __future__ import annotations

import base64
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FPP_DETECT_URL = os.environ.get(
    "FPP_API_URL", "https://api-us.faceplusplus.com/facepp/v3/detect"
)
FPP_ATTRIBUTES = (
    "gender,age,smiling,headpose,eyestatus,emotion,ethnicity,"
    "eyegaze,beauty,mouthstatus,blur,facequality,skinstatus"
)

FACE_OVAL_LANDMARK_NAMES = (
    *(f"contour_left{i}" for i in range(1, 17)),
    "contour_chin",
    *(f"contour_right{i}" for i in range(16, 0, -1)),
)
TOP_MARGIN = 0.30
BOTTOM_MARGIN = 0.20
MIN_SIDE_FROM_WIDTH = 1.22


@dataclass(frozen=True)
class PreprocessedPortrait:
    background_removed: np.ndarray
    framed: np.ndarray
    cropped: np.ndarray
    grayscale: np.ndarray
    foreground_fraction: float


class FacePlusPlusError(RuntimeError):
    """Raised when Face++ rejects a request or returns unusable data."""


def Get_Flattened_Dict(
    data: Mapping[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """Flatten a nested Face++ dictionary into one level."""
    items: list[tuple[str, Any]] = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.extend(Get_Flattened_Dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def Get_Euclidean_Distance(
    source_representation: Any, test_representation: Any
) -> float:
    """Return Euclidean distance between two points or vectors."""
    source = np.asarray(source_representation, dtype=float)
    target = np.asarray(test_representation, dtype=float)
    return float(np.linalg.norm(source - target))


def load_api_keys(location: str | Path = ".env") -> tuple[str, str]:
    """Load the single Study 2 Face++ credential pair without printing it."""
    from dotenv import dotenv_values

    config = dotenv_values(location)
    key = os.environ.get("FPP_KEY") or config.get("FPP_KEY")
    secret = os.environ.get("FPP_SECRET") or config.get("FPP_SECRET")
    if not key or not secret:
        raise FacePlusPlusError(
            "No Face++ credentials were found. Set FPP_KEY and FPP_SECRET."
        )
    return str(key), str(secret)


def _encode_jpeg(image: np.ndarray) -> bytes:
    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("OpenCV could not encode the input image as JPEG.")
    return base64.b64encode(encoded.tobytes())


def _point(landmarks: Mapping[str, tuple[float, float]], name: str):
    value = landmarks.get(name)
    return None if value is None else np.asarray(value, dtype=float)


def _eye_center(
    landmarks: Mapping[str, tuple[float, float]], side: str
) -> tuple[float, float]:
    direct_names = (
        f"{side}_eye_center",
        f"{side}_eye_pupil",
    )
    for name in direct_names:
        value = _point(landmarks, name)
        if value is not None:
            return float(value[0]), float(value[1])

    corner_names = (
        f"{side}_eye_left_corner",
        f"{side}_eye_right_corner",
    )
    corners = [_point(landmarks, name) for name in corner_names]
    if all(point is not None for point in corners):
        center = np.mean(np.stack(corners), axis=0)
        return float(center[0]), float(center[1])

    pupil_points = [
        np.asarray(value, dtype=float)
        for name, value in landmarks.items()
        if name.startswith(f"{side}_eye_") and "eyebrow" not in name
    ]
    if pupil_points:
        center = np.mean(np.stack(pupil_points), axis=0)
        return float(center[0]), float(center[1])

    raise FacePlusPlusError(f"Face++ response has no usable {side}-eye landmarks.")


def get_eye_centers(
    landmarks: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return robust left/right eye centers from Face++ landmark names."""
    return _eye_center(landmarks, "left"), _eye_center(landmarks, "right")


def scale_landmarks(
    landmarks: Mapping[str, tuple[float, float]],
    scale_x: float,
    scale_y: float,
) -> dict[str, tuple[float, float]]:
    """Map landmarks from an API-sized image back to the source image."""
    return {
        name: (float(point[0]) * scale_x, float(point[1]) * scale_y)
        for name, point in landmarks.items()
    }


def get_face_oval_points(
    landmarks: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Return the Face++ face-contour points used for dynamic cropping."""
    points = [
        np.asarray(landmarks[name], dtype=np.float64)
        for name in FACE_OVAL_LANDMARK_NAMES
        if name in landmarks
    ]
    if len(points) < 5:
        raise FacePlusPlusError("Face++ response has no usable face contour.")
    return np.stack(points)


def _fit_square_to_source(
    left: float, top: float, side: float, width: int, height: int
) -> tuple[int, int, int, int]:
    side_i = int(round(min(side, width, height)))
    if side_i < 1:
        raise ValueError("The calculated face crop is empty.")
    left_i = min(max(0, int(round(left))), width - side_i)
    top_i = min(max(0, int(round(top))), height - side_i)
    return left_i, top_i, left_i + side_i, top_i + side_i


def _dynamic_crop_box(
    points: np.ndarray,
    width: int,
    height: int,
    top_margin: float,
    bottom_margin: float,
    min_side_from_width: float,
) -> tuple[int, int, int, int]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    oval_width = float(x_max - x_min)
    oval_height = float(y_max - y_min)
    desired_top = float(y_min) - top_margin * oval_height
    desired_bottom = float(y_max) + bottom_margin * oval_height
    side = max(desired_bottom - desired_top, min_side_from_width * oval_width)
    center_x = float(x_min + x_max) / 2.0
    center_y = (desired_top + desired_bottom) / 2.0
    return _fit_square_to_source(
        center_x - side / 2.0,
        center_y - side / 2.0,
        side,
        width,
        height,
    )


def get_landmark_aligned_crop(
    img: np.ndarray,
    landmarks: Mapping[str, tuple[float, float]],
    output_size: int = 224,
    top_margin: float = TOP_MARGIN,
    bottom_margin: float = BOTTOM_MARGIN,
    min_side_from_width: float = MIN_SIDE_FROM_WIDTH,
    return_details: bool = False,
):
    """Level the eyes, dynamically crop the face oval, and resize to a square.

    Unlike the legacy affine transform, this does not force the eyes to fixed
    output coordinates. It rotates around the eye midpoint, derives a padded
    square from the aligned Face++ contour, then resizes that square to 224.
    """
    image = np.asarray(img)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an H x W x 3 color image.")
    height, width = image.shape[:2]
    eye_a, eye_b = (
        np.asarray(point, dtype=np.float64) for point in get_eye_centers(landmarks)
    )
    if eye_a[0] > eye_b[0]:
        eye_a, eye_b = eye_b, eye_a
    delta = eye_b - eye_a
    if np.linalg.norm(delta) < 1e-6:
        raise FacePlusPlusError("The two eye centers overlap.")
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    midpoint = tuple(((eye_a + eye_b) / 2.0).tolist())

    edge_width = max(1, min(8, height, width))
    edge_pixels = np.concatenate(
        (
            image[:edge_width, :, :3].reshape(-1, 3),
            image[-edge_width:, :, :3].reshape(-1, 3),
            image[:, :edge_width, :3].reshape(-1, 3),
            image[:, -edge_width:, :3].reshape(-1, 3),
        ),
        axis=0,
    )
    fill = tuple(int(value) for value in np.median(edge_pixels, axis=0))
    border_value = fill if image.shape[2] == 3 else (*fill, 0)
    transform = cv2.getRotationMatrix2D(midpoint, angle, 1.0)
    aligned = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    original_oval = get_face_oval_points(landmarks)
    homogeneous = np.column_stack((original_oval, np.ones(len(original_oval))))
    aligned_oval = homogeneous @ transform.T
    crop_box = _dynamic_crop_box(
        aligned_oval,
        width,
        height,
        top_margin,
        bottom_margin,
        min_side_from_width,
    )
    left, top, right, bottom = crop_box
    cropped = aligned[top:bottom, left:right]
    resized = cv2.resize(
        cropped,
        (output_size, output_size),
        interpolation=cv2.INTER_LANCZOS4,
    )

    aligned_corners = np.asarray(
        [(left, top), (right, top), (right, bottom), (left, bottom)],
        dtype=np.float64,
    )
    inverse = cv2.invertAffineTransform(transform)
    corner_homogeneous = np.column_stack(
        (aligned_corners, np.ones(len(aligned_corners)))
    )
    original_frame = corner_homogeneous @ inverse.T
    details = {
        "rotation_angle_degrees": float(angle),
        "crop_box_xyxy": crop_box,
        "original_display_frame_corners": original_frame,
        "original_face_oval_points": original_oval,
        "top_margin_ratio": float(top_margin),
        "bottom_margin_ratio": float(bottom_margin),
        "min_side_from_width": float(min_side_from_width),
    }
    return (resized, details) if return_details else resized


def draw_crop_frame_and_contour(
    img: np.ndarray,
    frame_corners: np.ndarray,
    contour_points: np.ndarray,
) -> np.ndarray:
    """Draw the trial's blue crop frame and purple contour on the source."""
    result = np.asarray(img).copy()
    height, width = result.shape[:2]
    line_width = max(4, round(min(width, height) / 180))
    radius = max(3, round(min(width, height) / 220))
    polygon = np.rint(frame_corners).astype(np.int32).reshape((-1, 1, 2))
    line_color = (204, 102, 0, 255) if result.shape[2] == 4 else (204, 102, 0)
    point_color = (255, 99, 108, 255) if result.shape[2] == 4 else (255, 99, 108)
    cv2.polylines(
        result,
        [polygon],
        isClosed=True,
        color=line_color,
        thickness=line_width,
        lineType=cv2.LINE_AA,
    )
    for x, y in np.rint(contour_points).astype(np.int32):
        cv2.circle(
            result,
            (int(x), int(y)),
            radius,
            color=point_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return result


class PersonSegmenter:
    """Reusable MediaPipe person segmenter for complete raw portraits."""

    def __init__(
        self,
        model_path: str | Path,
        lower_confidence: float = 0.12,
        upper_confidence: float = 0.78,
    ) -> None:
        self.model_path = Path(model_path)
        self.lower_confidence = float(lower_confidence)
        self.upper_confidence = float(upper_confidence)
        self._mp = None
        self._segmenter = None
        if not self.model_path.exists():
            raise FileNotFoundError("The person-segmentation model was not found.")
        if not 0 <= self.lower_confidence < self.upper_confidence <= 1:
            raise ValueError(
                "Segmentation confidence bounds must satisfy 0 <= lower < upper <= 1."
            )

    def _open(self) -> None:
        if self._segmenter is not None:
            return
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for background removal. Install project requirements."
            ) from exc
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._mp = mp
        self._segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment(self, img: np.ndarray) -> np.ndarray:
        """Return a soft float32 person alpha mask in the range [0, 1]."""
        image = np.asarray(img)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected an H x W x 3 color image for segmentation.")
        self._open()
        rgb = np.ascontiguousarray(
            cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
        )
        result = self._segmenter.segment(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.confidence_masks:
            raise ValueError("Person segmentation returned no confidence mask.")
        confidence = np.asarray(
            result.confidence_masks[0].numpy_view(), dtype=np.float32
        ).copy()
        return self._refine(confidence, image.shape[:2])

    def _refine(
        self, confidence: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        height, width = image_shape
        if confidence.shape != image_shape:
            confidence = cv2.resize(
                confidence, (width, height), interpolation=cv2.INTER_LINEAR
            )
        alpha = np.clip(
            (confidence - self.lower_confidence)
            / (self.upper_confidence - self.lower_confidence),
            0,
            1,
        )
        alpha = alpha * alpha * (3 - 2 * alpha)
        return np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.8), 0, 1).astype(
            np.float32
        )

    def close(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def get_person_alpha(
    img: np.ndarray,
    model_path: str | Path,
    lower_confidence: float = 0.12,
    upper_confidence: float = 0.78,
) -> np.ndarray:
    """One-shot convenience wrapper around :class:`PersonSegmenter`."""
    with PersonSegmenter(
        model_path,
        lower_confidence=lower_confidence,
        upper_confidence=upper_confidence,
    ) as segmenter:
        return segmenter.segment(img)


def apply_person_alpha(
    img: np.ndarray, alpha: np.ndarray, hidden_background_value: int = 238
) -> np.ndarray:
    """Attach person alpha as BGRA and erase RGB under transparent pixels.

    Erasing hidden RGB prevents applications that ignore PNG alpha from
    revealing the original background.
    """
    image = np.asarray(img)
    mask = np.asarray(alpha, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3 or mask.shape != image.shape[:2]:
        raise ValueError("Image and alpha dimensions do not match.")
    if not 0 <= hidden_background_value <= 255:
        raise ValueError("hidden_background_value must be between 0 and 255.")
    alpha_u8 = np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8)
    visible_bgr = image[..., :3].copy()
    visible_bgr[alpha_u8 == 0] = hidden_background_value
    return np.dstack((visible_bgr, alpha_u8))


def composite_transparency(
    img: np.ndarray, background_value: int = 238
) -> np.ndarray:
    """Composite BGRA onto a neutral BGR background for stable grayscale."""
    image = np.asarray(img)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected an H x W x 4 BGRA image.")
    if not 0 <= background_value <= 255:
        raise ValueError("background_value must be between 0 and 255.")
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    composed = (
        image[..., :3].astype(np.float32) * alpha
        + float(background_value) * (1.0 - alpha)
    )
    return np.rint(composed).astype(np.uint8)


def preprocess_portrait(
    img: np.ndarray,
    landmarks: Mapping[str, tuple[float, float]],
    segmenter: PersonSegmenter,
    grayscale_background: int = 238,
) -> PreprocessedPortrait:
    """Run background removal, framing, cropping, and grayscale conversion."""
    image = np.asarray(img)
    person_alpha = segmenter.segment(image)
    background_removed = apply_person_alpha(
        image,
        person_alpha,
        hidden_background_value=grayscale_background,
    )
    cropped, crop_details = get_landmark_aligned_crop(
        background_removed,
        landmarks,
        return_details=True,
    )
    framed = draw_crop_frame_and_contour(
        background_removed,
        crop_details["original_display_frame_corners"],
        crop_details["original_face_oval_points"],
    )
    grayscale_source = composite_transparency(
        cropped,
        background_value=grayscale_background,
    )
    grayscale = np.rint(
        to_grayscale(grayscale_source, bgr=True, vgg=False) * 255.0
    ).astype(np.uint8)
    return PreprocessedPortrait(
        background_removed=background_removed,
        framed=framed,
        cropped=cropped,
        grayscale=grayscale,
        foreground_fraction=float((person_alpha >= 0.5).mean()),
    )


def _quality_score(
    landmarks: Mapping[str, tuple[float, float]], attributes: Mapping[str, Any]
) -> int:
    """Reproduce the original repository's five-part usability score."""
    left_eye, right_eye = get_eye_centers(landmarks)
    eye_distance = Get_Euclidean_Distance(left_eye, right_eye)
    mouth_ok = float(attributes.get("mouthstatus_other_occlusion", 100)) <= 50
    left_eye_ok = float(attributes.get("eyestatus_left_eye_status_occlusion", 100)) <= 50
    right_eye_ok = float(attributes.get("eyestatus_right_eye_status_occlusion", 100)) <= 50
    pitch_ok = abs(float(attributes.get("headpose_pitch_angle", 999))) <= 10
    yaw_ok = abs(float(attributes.get("headpose_yaw_angle", 999))) <= 15
    size_ok = eye_distance >= 40
    return int(mouth_ok) + int(left_eye_ok or right_eye_ok) + int(pitch_ok) + int(yaw_ok) + int(size_ok)


def Get_FacePlusPlus_Outputs(
    img: np.ndarray,
    FPP_KEY: str,
    FPP_SECRET: str,
    landmark_106: int = 2,
    compare_face: int = 1,
    timeout: float = 15.0,
) -> tuple[int, dict[str, tuple[float, float]], dict[str, Any], dict[str, Any]]:
    """Call Face++ and return the largest usable face.

    This keeps the legacy function name and four-value return shape, but unlike
    the original implementation it raises a descriptive exception instead of
    silently turning every API/network error into four zeroes.
    """
    payload = {
        "api_key": FPP_KEY,
        "api_secret": FPP_SECRET,
        "image_base64": _encode_jpeg(img),
        "return_landmark": landmark_106,
        "return_attributes": FPP_ATTRIBUTES,
    }
    import requests

    try:
        response = requests.post(FPP_DETECT_URL, data=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        raise FacePlusPlusError(f"Face++ request failed: {exc}") from exc
    except ValueError as exc:
        raise FacePlusPlusError("Face++ returned invalid JSON.") from exc

    if body.get("error_message"):
        raise FacePlusPlusError(f"Face++ error: {body['error_message']}")
    faces = body.get("faces") or []
    if not faces:
        raise FacePlusPlusError("Face++ detected no face.")
    if len(faces) > compare_face:
        raise FacePlusPlusError(
            f"Face++ detected {len(faces)} faces; maximum allowed is {compare_face}."
        )

    parsed: list[tuple[float, int, dict, dict, dict]] = []
    for face in faces[:5]:
        raw_landmarks = face.get("landmark") or {}
        landmarks = {
            name: (float(value["x"]), float(value["y"]))
            for name, value in raw_landmarks.items()
            if isinstance(value, Mapping) and "x" in value and "y" in value
        }
        attributes = Get_Flattened_Dict(face.get("attributes") or {})
        left_eye, right_eye = get_eye_centers(landmarks)
        eye_distance = Get_Euclidean_Distance(left_eye, right_eye)
        score = _quality_score(landmarks, attributes)
        parsed.append((eye_distance, score, landmarks, attributes, face.get("face_rectangle") or {}))

    _, score, landmarks, attributes, rectangle = max(parsed, key=lambda item: item[0])
    return score, landmarks, attributes, rectangle


def get_rectangle(rectangle: Mapping[str, Any]) -> list[list[tuple[int, int]]]:
    """Convert a Face++ rectangle to top and bottom corner pairs."""
    left = int(rectangle["left"])
    top = int(rectangle["top"])
    width = int(rectangle["width"])
    height = int(rectangle["height"])
    return [
        [(left, top), (left + width, top)],
        [(left, top + height), (left + width, top + height)],
    ]


def get_faceplusplus_outputs(
    img: np.ndarray,
    credentials: tuple[str, str],
    verify_score: int = 0,
    max_faces: int = 1,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Call Face++ once with the configured Study 2 credential pair."""
    key, secret = credentials
    score, landmarks, attributes, rectangle = Get_FacePlusPlus_Outputs(
        img,
        key,
        secret,
        landmark_106=2,
        compare_face=max_faces,
        timeout=timeout,
    )
    if score < verify_score:
        raise FacePlusPlusError(
            f"face usability score {score}/5 is below required {verify_score}/5"
        )
    return {
        "score": score,
        "landmarks": landmarks,
        "attributes": attributes,
        "rectangle": dict(rectangle),
        "credential": "FPP",
    }


def get_rotated_image(
    img: np.ndarray,
    src_lt: tuple[float, float],
    src_rt: tuple[float, float],
    dst_lt: tuple[float, float] = (33, 33),
    dst_rt: tuple[float, float] = (191, 33),
    imsize: int = 224,
) -> np.ndarray:
    """Align, scale, and place a face using its two eye centers."""
    in_points = [tuple(src_lt), tuple(src_rt)]
    out_points = [tuple(dst_lt), tuple(dst_rt)]
    sin60 = math.sin(math.radians(60))
    cos60 = math.cos(math.radians(60))

    def third_point(points: list[tuple[float, float]]) -> tuple[float, float]:
        x = cos60 * (points[0][0] - points[1][0]) - sin60 * (
            points[0][1] - points[1][1]
        ) + points[1][0]
        y = sin60 * (points[0][0] - points[1][0]) + cos60 * (
            points[0][1] - points[1][1]
        ) + points[1][1]
        return x, y

    in_points.append(third_point(in_points))
    out_points.append(third_point(out_points))
    transform = cv2.getAffineTransform(
        np.asarray(in_points, dtype=np.float32),
        np.asarray(out_points, dtype=np.float32),
    )
    return cv2.warpAffine(
        img,
        transform,
        (imsize, imsize),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def to_grayscale(img: np.ndarray, bgr: bool = True, vgg: bool = True) -> np.ndarray:
    """Convert an image to normalized float32 grayscale in the range [0, 1].

    The original function clipped uint8 input directly to [0, 1], which made
    most pixels white.  This version first normalizes 0..255 input.  With
    ``vgg=True`` the grayscale plane is repeated into three channels; with
    ``vgg=False`` a single 2-D plane is returned.
    """
    array = np.asarray(img)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an H x W x 3 color image.")
    work = array[..., :3].astype(np.float32)
    if work.max(initial=0) > 1.0:
        work /= 255.0
    weights = np.asarray(
        [0.1140, 0.5870, 0.2989] if bgr else [0.2989, 0.5870, 0.1140],
        dtype=np.float32,
    )
    gray = np.clip(np.dot(work, weights), 0.0, 1.0).astype(np.float32)
    return np.repeat(gray[..., None], 3, axis=2) if vgg else gray
