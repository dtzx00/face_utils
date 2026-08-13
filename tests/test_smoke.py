import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_utils import (
    Get_Euclidean_Distance,
    Get_Flattened_Dict,
    apply_person_alpha,
    composite_transparency,
    draw_crop_frame_and_contour,
    get_eye_centers,
    get_landmark_aligned_crop,
    get_rotated_image,
    preprocess_portrait,
    scale_landmarks,
    to_grayscale,
)
from process_study2 import (
    append_manifest,
    group_images_by_person,
    is_terminal_photo_error,
    manifest_input_path,
    manifest_output_path,
    normalize_id,
    prepare_faceplusplus_image,
)


class UtilsTests(unittest.TestCase):
    @staticmethod
    def synthetic_landmarks():
        names = [
            *(f"contour_left{i}" for i in range(1, 17)),
            "contour_chin",
            *(f"contour_right{i}" for i in range(16, 0, -1)),
        ]
        landmarks = {
            name: (
                110 + 48 * np.cos(2 * np.pi * index / len(names)),
                115 + 62 * np.sin(2 * np.pi * index / len(names)),
            )
            for index, name in enumerate(names)
        }
        landmarks.update(
            {
                "left_eye_center": (78, 82),
                "right_eye_center": (142, 92),
            }
        )
        return landmarks

    def test_study2_id_suffixes_remain_distinct(self):
        self.assertEqual(normalize_id("SUBJECT001"), "SUBJECT001")
        for marked in (
            "SUBJECT001STAR",
            "SUBJECT001_STAR",
            "SUBJECT001*",
            "SUBJECT001_",
            "SUBJECT001-",
        ):
            self.assertEqual(normalize_id(marked), "SUBJECT001_")
        self.assertEqual(normalize_id("SUBJECT001**"), "SUBJECT001__")
        self.assertEqual(normalize_id("SUBJECT001---"), "SUBJECT001___")
        self.assertNotEqual(
            normalize_id("SUBJECT002*"), normalize_id("SUBJECT002**")
        )

    def test_manifest_paths_are_portable(self):
        output_root = Path("results")
        self.assertEqual(
            manifest_input_path("input_01", Path("group") / "portrait.jpg"),
            "input_01/5a8551d24c912368a506.jpg",
        )
        self.assertEqual(
            manifest_output_path(output_root, output_root / "grayscale" / "portrait.png"),
            "grayscale/portrait.png",
        )

    def test_flatten(self):
        self.assertEqual(Get_Flattened_Dict({"a": {"b": 2}}), {"a_b": 2})

    def test_distance(self):
        self.assertEqual(Get_Euclidean_Distance((0, 0), (3, 4)), 5.0)

    def test_eye_centers_from_corners(self):
        landmarks = {
            "left_eye_left_corner": (10, 20),
            "left_eye_right_corner": (20, 20),
            "right_eye_left_corner": (40, 20),
            "right_eye_right_corner": (50, 20),
        }
        self.assertEqual(get_eye_centers(landmarks), ((15.0, 20.0), (45.0, 20.0)))

    def test_alignment_shape(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        aligned = get_rotated_image(image, (40, 50), (100, 50))
        self.assertEqual(aligned.shape, (224, 224, 3))

    def test_dynamic_landmark_crop_and_annotation(self):
        image = np.full((240, 220, 3), 230, dtype=np.uint8)
        crop, details = get_landmark_aligned_crop(
            image, self.synthetic_landmarks(), return_details=True
        )
        self.assertEqual(crop.shape, (224, 224, 3))
        left, top, right, bottom = details["crop_box_xyxy"]
        self.assertEqual(right - left, bottom - top)
        self.assertGreater(details["rotation_angle_degrees"], 0)
        framed = draw_crop_frame_and_contour(
            image,
            details["original_display_frame_corners"],
            details["original_face_oval_points"],
        )
        self.assertEqual(framed.shape, image.shape)
        self.assertFalse(np.array_equal(framed, image))

    def test_transparency_survives_frame_and_crop(self):
        image = np.full((240, 220, 3), 180, dtype=np.uint8)
        yy, xx = np.ogrid[:240, :220]
        alpha = (
            ((xx - 110) / 75) ** 2 + ((yy - 120) / 95) ** 2 <= 1
        ).astype(np.float32)
        transparent = apply_person_alpha(image, alpha)
        crop, details = get_landmark_aligned_crop(
            transparent, self.synthetic_landmarks(), return_details=True
        )
        framed = draw_crop_frame_and_contour(
            transparent,
            details["original_display_frame_corners"],
            details["original_face_oval_points"],
        )
        self.assertEqual(crop.shape, (224, 224, 4))
        self.assertEqual(framed.shape, (240, 220, 4))
        self.assertTrue(np.any(crop[..., 3] == 0))
        self.assertTrue(np.any(framed[..., 3] == 255))

    def test_transparency_composites_to_neutral_background(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = (10, 20, 30)
        alpha = np.zeros((2, 2), dtype=np.float32)
        alpha[0, 0] = 1.0
        transparent = apply_person_alpha(image, alpha)
        self.assertTrue(np.array_equal(transparent[1, 1, :3], (238, 238, 238)))
        composite = composite_transparency(transparent, background_value=238)
        self.assertTrue(np.array_equal(composite[0, 0], (10, 20, 30)))
        self.assertTrue(np.array_equal(composite[1, 1], (238, 238, 238)))

    def test_complete_preprocessing_ends_at_grayscale(self):
        image = np.full((240, 220, 3), 180, dtype=np.uint8)

        class Segmenter:
            @staticmethod
            def segment(value):
                return np.ones(value.shape[:2], dtype=np.float32)

        result = preprocess_portrait(
            image,
            self.synthetic_landmarks(),
            Segmenter(),
        )
        self.assertEqual(result.background_removed.shape, (240, 220, 4))
        self.assertEqual(result.framed.shape, (240, 220, 4))
        self.assertEqual(result.cropped.shape, (224, 224, 4))
        self.assertEqual(result.grayscale.shape, (224, 224))
        self.assertEqual(result.grayscale.dtype, np.uint8)
        self.assertEqual(result.foreground_fraction, 1.0)

    def test_landmarks_can_be_mapped_back_to_original_resolution(self):
        scaled = scale_landmarks({"left_eye_center": (10, 20)}, 2.0, 3.0)
        self.assertEqual(scaled["left_eye_center"], (20.0, 60.0))

    def test_uint8_grayscale_not_clipped_white(self):
        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        gray = to_grayscale(image, vgg=False)
        self.assertEqual(gray.shape, (4, 4))
        self.assertTrue(np.allclose(gray, 128 / 255, atol=1e-3))
        self.assertEqual(to_grayscale(image, vgg=True).shape, (4, 4, 3))

    def test_oversized_api_image_is_downscaled(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(1800, 2400, 3), dtype=np.uint8)
        prepared = prepare_faceplusplus_image(image, max_jpeg_bytes=250_000)
        self.assertLess(prepared.shape[0], image.shape[0])
        self.assertLess(prepared.shape[1], image.shape[1])

    def test_large_dimensions_are_downscaled_even_when_jpeg_is_small(self):
        image = np.zeros((5000, 1800, 3), dtype=np.uint8)
        prepared = prepare_faceplusplus_image(image)
        self.assertLessEqual(max(prepared.shape[:2]), 4096)

    def test_candidates_are_grouped_by_person_in_discovery_order(self):
        images = [
            ("source", "ID1", "first", "first"),
            ("source", "ID2", "only", "only"),
            ("source", "ID1", "second", "second"),
        ]
        self.assertEqual(
            group_images_by_person(images),
            [("ID1", [images[0], images[2]]), ("ID2", [images[1]])],
        )

    def test_only_photo_level_failures_are_terminal(self):
        self.assertTrue(is_terminal_photo_error("Face++ detected no face."))
        self.assertTrue(
            is_terminal_photo_error("Face++ detected 3 faces; maximum allowed is 1.")
        )
        self.assertTrue(is_terminal_photo_error("OpenCV could not decode the image."))
        self.assertFalse(is_terminal_photo_error("403 Client Error: Forbidden"))
        self.assertFalse(is_terminal_photo_error("Read timed out"))

    def test_old_manifest_header_remains_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["input_path", "status"])
                writer.writeheader()
                writer.writerow({"input_path": "old.jpg", "status": "ok"})
            append_manifest(
                manifest,
                {
                    "input_path": "new.jpg",
                    "status": "ok",
                    "output_framed": "new_framed.jpg",
                },
            )
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1], {"input_path": "new.jpg", "status": "ok"})


if __name__ == "__main__":
    unittest.main()
