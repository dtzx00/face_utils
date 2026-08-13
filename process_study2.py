"""Batch Face++ alignment and grayscale conversion for Study 2 photos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

DEFAULT_SEGMENTATION_MODEL = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".avif",
}
MANIFEST_FIELDS = [
    "input_path",
    "source",
    "person_id",
    "output_raw",
    "output_background_removed",
    "output_framed",
    "output_grayscale",
    "output_cropped",
    "status",
    "quality_score",
    "segmentation_foreground_fraction",
    "credential",
    "error",
    "rectangle_json",
    "landmarks_json",
    "attributes_json",
]


def normalize_id(value: object) -> str:
    text = str(value or "").strip().upper()
    suffix_count = 0
    while match := re.search(r"(?:_?STAR|[*_-])$", text):
        suffix_count += 1
        text = text[: match.start()]
    normalized = re.sub(r"[^0-9A-Z]", "", text)
    return f"{normalized}{'_' * suffix_count}" if normalized else ""


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{namespace}t"))
    value = cell.find(f"{namespace}v")
    if value is None or value.text is None:
        return ""
    if cell.get("t") == "s":
        return shared_strings[int(value.text)]
    return value.text


def _read_xlsx_records(path: Path) -> list[dict[str, str]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    namespace = f"{{{main_ns}}}"
    with zipfile.ZipFile(path) as workbook_zip:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{namespace}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{namespace}t"))
                )

        workbook_root = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        sheets = workbook_root.find(f"{namespace}sheets")
        if sheets is None or not list(sheets):
            raise ValueError("The ID workbook contains no worksheets.")
        sheet = next(
            (item for item in sheets if item.get("name", "").casefold() == "merged"),
            list(sheets)[0],
        )
        relationship_id = sheet.get(f"{{{rel_ns}}}id")
        relationships = ET.fromstring(
            workbook_zip.read("xl/_rels/workbook.xml.rels")
        )
        relationship = next(
            (
                item
                for item in relationships.findall(f"{{{package_rel_ns}}}Relationship")
                if item.get("Id") == relationship_id
            ),
            None,
        )
        if relationship is None:
            raise ValueError("The ID workbook has an invalid worksheet relationship.")
        target = relationship.get("Target", "worksheets/sheet1.xml").lstrip("/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheet_root = ET.fromstring(workbook_zip.read(sheet_path))

    rows = sheet_root.findall(f".//{namespace}sheetData/{namespace}row")
    if not rows:
        return []
    header_by_column: dict[str, str] = {}
    for cell in rows[0].findall(f"{namespace}c"):
        column_match = re.match(r"[A-Z]+", cell.get("r", ""))
        if column_match:
            header_by_column[column_match.group()] = _xlsx_cell_value(
                cell, shared_strings
            )
    wanted_headers = {"id", "face_source_file", "face_person_id_raw"}
    selected_columns = {
        column: header.strip().casefold()
        for column, header in header_by_column.items()
        if header.strip().casefold() in wanted_headers
    }
    if "id" not in selected_columns.values():
        raise ValueError(f"No ID column found in {path}")

    records: list[dict[str, str]] = []
    for row in rows[1:]:
        row_number = row.get("r", "")
        record: dict[str, str] = {}
        for column, header in selected_columns.items():
            cell = row.find(f"{namespace}c[@r='{column}{row_number}']")
            record[header] = (
                _xlsx_cell_value(cell, shared_strings) if cell is not None else ""
            )
        records.append(record)
    return records


def _source_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    period = re.search(r"(?:week|wave|round)(\d+)", normalized)
    return period.group(0) if period else normalized


def _face_source_collections(source_file: str, raw_id: str) -> set[str]:
    parts = [
        part
        for value in (source_file, raw_id)
        for part in value.replace("\\", "/").strip("/").split("/")
        if part
    ]
    return {key for part in parts if (key := _source_key(part))}


def _face_raw_alias(raw_id: str) -> str:
    leaf = raw_id.replace("\\", "/").rstrip("/").split("/")[-1]
    leaf = re.sub(r"\(\d+\)$", "", leaf.strip())
    return normalize_id(leaf)


def read_study2_id_lookup(
    path: Path | None,
) -> tuple[set[str], dict[tuple[str, str], str], dict[str, str]]:
    if path is None:
        return set(), {}, {}
    if not path.exists():
        raise FileNotFoundError("The configured ID list was not found.")
    if path.suffix.casefold() != ".xlsx":
        return read_study2_ids(path), {}, {}

    records = _read_xlsx_records(path)
    ids: set[str] = set()
    source_targets: dict[tuple[str, str], set[str]] = {}
    global_targets: dict[str, set[str]] = {}
    raw_by_normalized: dict[str, set[str]] = {}
    for record in records:
        raw_official = record.get("id", "")
        official = normalize_id(raw_official)
        if not official:
            continue
        ids.add(official)
        raw_by_normalized.setdefault(official, set()).add(raw_official.strip())
        global_targets.setdefault(official, set()).add(official)

        raw_id = record.get("face_person_id_raw", "")
        alias = _face_raw_alias(raw_id) if raw_id else ""
        if not alias:
            continue
        global_targets.setdefault(alias, set()).add(official)
        collections = _face_source_collections(
            record.get("face_source_file", ""), raw_id
        )
        for collection in collections:
            source_targets.setdefault((collection, alias), set()).add(official)

    collisions = {
        normalized: values
        for normalized, values in raw_by_normalized.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(
            f"The ID list contains {len(collisions)} normalization collision(s)."
        )

    source_aliases = {
        key: next(iter(targets))
        for key, targets in source_targets.items()
        if len(targets) == 1
    }
    global_aliases = {
        alias: next(iter(targets))
        for alias, targets in global_targets.items()
        if len(targets) == 1
    }
    return ids, source_aliases, global_aliases


def read_study2_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError("The configured ID list was not found.")
    if path.suffix.casefold() == ".xlsx":
        return read_study2_id_lookup(path)[0]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            id_column = next(
                (
                    name
                    for name in ("ID", "id", "person_id", "code", "code_excel", "code_raw")
                    if name in columns
                ),
                None,
            )
            if id_column is None:
                raise ValueError(f"No person-ID column found in {path}")
            raw_ids = [row.get(id_column, "") for row in reader]

    ids: set[str] = set()
    raw_by_normalized: dict[str, set[str]] = {}
    for raw_id in raw_ids:
        person_id = normalize_id(raw_id)
        if not person_id:
            continue
        ids.add(person_id)
        raw_by_normalized.setdefault(person_id, set()).add(str(raw_id).strip())
    collisions = {
        normalized: values
        for normalized, values in raw_by_normalized.items()
        if len(values) > 1
    }
    if collisions:
        raise ValueError(
            f"The ID list contains {len(collisions)} normalization collision(s)."
        )
    return ids


def default_input_roots(
    study2_root: Path, fallback_root: Path | None = None
) -> list[tuple[str, Path]]:
    if not study2_root.exists():
        raise FileNotFoundError("The configured portrait root was not found.")
    roots: list[tuple[str, Path]] = []
    seen_roots: set[str] = set()
    for path in sorted(study2_root.glob("Raw - *")):
        if path.is_dir():
            resolved = path.resolve()
            roots.append((f"priority_{len(roots) + 1:02d}", resolved))
            seen_roots.add(str(resolved).casefold())

    if fallback_root is not None and fallback_root.is_dir():
        for path in sorted(fallback_root.iterdir()):
            if not path.is_dir():
                continue
            name_folded = path.name.casefold()
            if name_folded in {"analysis", "output", "outputs"}:
                continue
            resolved = path.resolve()
            key = str(resolved).casefold()
            if key in seen_roots:
                continue
            roots.append((f"fallback_{len(roots) + 1:02d}", resolved))
            seen_roots.add(key)
    if not roots:
        raise FileNotFoundError("No 'Raw - *' folders were found in the portrait root.")
    return roots


def _candidate_ids(path: Path, root: Path | None = None) -> list[str]:
    directory_ids: list[str] = []
    current = path.parent
    while True:
        candidate = normalize_id(current.name)
        if candidate:
            directory_ids.append(candidate)
        if root is None or current == root or current.parent == current:
            break
        current = current.parent
    stem = path.stem.upper()
    stem = re.sub(r"^(\d+)[_-]", "", stem)
    stem = re.sub(r"[_-](\d+)$", "", stem)
    filename = normalize_id(stem)
    return list(dict.fromkeys(value for value in (*directory_ids, filename) if value))


def discover_images(
    roots: list[tuple[str, Path]],
    allowed_ids: set[str],
    max_images_per_person: int,
    source_aliases: dict[tuple[str, str], str] | None = None,
    global_aliases: dict[str, str] | None = None,
) -> list[tuple[str, str, Path, Path]]:
    selected: list[tuple[str, str, Path, Path]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    source_aliases = source_aliases or {}
    global_aliases = global_aliases or {}
    for source, root in roots:
        root_key = _source_key(root.name)
        for image_path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image_path.name.startswith("._"):
                continue
            resolved = str(image_path.resolve()).lower()
            if resolved in seen:
                continue
            candidates = _candidate_ids(image_path, root)
            if allowed_ids:
                person_id = next(
                    (
                        value
                        if value in allowed_ids
                        else source_aliases.get((root_key, value), "")
                        or global_aliases.get(value, "")
                        for value in candidates
                        if value in allowed_ids
                        or (root_key, value) in source_aliases
                        or value in global_aliases
                    ),
                    "",
                )
                if not person_id:
                    continue
            else:
                person_id = candidates[0] if candidates else "unknown"
            if max_images_per_person > 0 and counts.get(person_id, 0) >= max_images_per_person:
                continue
            seen.add(resolved)
            counts[person_id] = counts.get(person_id, 0) + 1
            try:
                relative = image_path.relative_to(root)
            except ValueError:
                relative = Path(image_path.name)
            selected.append((source, person_id, image_path, relative))
    return selected


def group_images_by_person(images):
    """Keep discovery priority while collecting every candidate for each ID."""
    grouped = OrderedDict()
    for item in images:
        grouped.setdefault(item[1], []).append(item)
    return list(grouped.items())


def read_image(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None and path.suffix.casefold() == ".avif":
        import pillow_avif
        from PIL import Image

        _ = pillow_avif
        with Image.open(path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image is None:
        raise ValueError("OpenCV could not decode the image.")
    return image


def write_image(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"OpenCV could not encode {extension} output.")
    encoded.tofile(path)


def prepare_faceplusplus_image(
    image, max_jpeg_bytes: int = 1_400_000, max_dimension: int = 4096
):
    """Fit Face++ byte and pixel-dimension limits; outputs remain 224x224."""
    import cv2

    prepared = image
    while True:
        ok, encoded = cv2.imencode(".jpg", prepared)
        if not ok:
            raise ValueError("OpenCV could not encode the Face++ input image.")
        largest_dimension = max(prepared.shape[:2])
        if encoded.nbytes <= max_jpeg_bytes and largest_dimension <= max_dimension:
            return prepared
        scale = 1.0
        if encoded.nbytes > max_jpeg_bytes:
            scale = min(scale, (max_jpeg_bytes / encoded.nbytes) ** 0.5 * 0.9)
        if largest_dimension > max_dimension:
            scale = min(scale, max_dimension / largest_dimension * 0.98)
        scale = min(scale, 0.9)
        new_width = max(224, int(round(prepared.shape[1] * scale)))
        new_height = max(224, int(round(prepared.shape[0] * scale)))
        if (new_width, new_height) == (prepared.shape[1], prepared.shape[0]):
            return prepared
        prepared = cv2.resize(
            prepared, (new_width, new_height), interpolation=cv2.INTER_AREA
        )


def load_completed(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["input_path"].lower()
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def is_terminal_photo_error(error: str) -> bool:
    """Return whether trying the same physical file again cannot help."""
    text = (error or "").casefold()
    return (
        "detected no face" in text
        or ("detected " in text and "faces; maximum allowed" in text)
        or "could not decode the image" in text
    )


def load_manifest_state(manifest: Path) -> tuple[set[str], set[str], set[str]]:
    """Load successful paths/people and photo-level terminal failures."""
    completed_paths: set[str] = set()
    completed_people: set[str] = set()
    terminal_paths: set[str] = set()
    if not manifest.exists():
        return completed_paths, completed_people, terminal_paths
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            input_key = (row.get("input_path") or "").lower()
            if row.get("status") == "ok":
                if input_key:
                    completed_paths.add(input_key)
                person_id = normalize_id(row.get("person_id", ""))
                if person_id:
                    completed_people.add(person_id)
            elif input_key and is_terminal_photo_error(row.get("error", "")):
                terminal_paths.add(input_key)
    return completed_paths, completed_people, terminal_paths


def append_manifest(manifest: Path, row: dict[str, object]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.exists() and manifest.stat().st_size > 0
    fieldnames = MANIFEST_FIELDS
    if exists:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header:
            fieldnames = existing_header
    with manifest.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def output_paths(output: Path, person_id: str, relative: Path):
    """Return five flat output paths without source/person subdirectories."""
    stem = relative.stem
    base_name = stem if normalize_id(stem).startswith(person_id) else f"{person_id}__{stem}"
    raw = output / "raw" / f"{base_name}{relative.suffix.lower()}"
    background_removed = output / "background_removed" / f"{base_name}.png"
    framed = output / "framed" / f"{base_name}.png"
    cropped = output / "cropped" / f"{base_name}.png"
    grayscale = output / "grayscale" / f"{base_name}.png"
    return raw, background_removed, framed, cropped, grayscale


def manifest_input_path(source: str, relative: Path) -> str:
    logical_path = (Path(source) / relative).as_posix()
    token = hashlib.sha256(logical_path.encode("utf-8")).hexdigest()[:20]
    return (Path(source) / f"{token}{relative.suffix.casefold()}").as_posix()


def manifest_output_path(output_root: Path, path: Path) -> str:
    return path.relative_to(output_root).as_posix()


def privacy_safe_error(exc: Exception) -> str:
    if isinstance(exc, OSError):
        message = exc.strerror or "filesystem operation failed"
    else:
        message = str(exc)
    return f"{type(exc).__name__}: {message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove portrait backgrounds, draw the Face++ frame, crop, and convert to grayscale for Study 2."
    )
    parser.add_argument(
        "--study2-root",
        type=Path,
        help="Directory containing the curated 'Raw - *' Study 2 folders.",
    )
    parser.add_argument(
        "--fallback-root",
        type=Path,
        help="Search this root after the curated Study 2 sources.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        help="Custom image root; may be repeated instead of using --study2-root.",
    )
    parser.add_argument(
        "--extra-input-root",
        type=Path,
        action="append",
        help="Append an extracted archive/cache directory after the default photo roots.",
    )
    parser.add_argument(
        "--ids-file",
        "--ids-csv",
        dest="ids_file",
        type=Path,
        help="CSV/XLSX used to restrict processing to Study 2 person IDs; use --no-id-filter to disable.",
    )
    parser.add_argument("--no-id-filter", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("study2_preprocessed_output"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--segmentation-model",
        type=Path,
        default=DEFAULT_SEGMENTATION_MODEL,
        help="MediaPipe person-segmentation .tflite model.",
    )
    parser.add_argument(
        "--grayscale-background",
        type=int,
        choices=range(0, 256),
        default=238,
        help="Neutral background value used for the final single-channel grayscale PNG.",
    )
    parser.add_argument(
        "--verify-score",
        type=int,
        choices=range(0, 6),
        default=0,
        help="Minimum custom 0-5 usability score; Study 2 defaults to 0 (record only, do not filter).",
    )
    parser.add_argument("--max-faces", type=int, default=1)
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=1,
        help="Candidate limit used with --first-photo-only; 0 means all images.",
    )
    parser.add_argument(
        "--first-photo-only",
        action="store_true",
        help="Disable Study 2 fallback behavior and select only the first candidate per ID.",
    )
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images; 0 means all.")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover inputs and print counts without uploading images or requiring API keys.",
    )
    parser.add_argument(
        "--show-identifiers",
        action="store_true",
        help="Include participant IDs and filenames in console output.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    if args.input_root:
        roots = [
            (f"input_{index:02d}", path.resolve())
            for index, path in enumerate(args.input_root, start=1)
        ]
    elif args.study2_root:
        roots = default_input_roots(args.study2_root, args.fallback_root)
    else:
        raise ValueError("Provide --study2-root or at least one --input-root.")
    if args.extra_input_root:
        roots.extend(
            (f"extra_{index:02d}", path.resolve())
            for index, path in enumerate(args.extra_input_root, start=1)
        )
    if args.no_id_filter:
        allowed_ids, source_aliases, global_aliases = set(), {}, {}
    else:
        if args.ids_file is None:
            raise ValueError("Provide --ids-file or use --no-id-filter.")
        allowed_ids, source_aliases, global_aliases = read_study2_id_lookup(
            args.ids_file
        )
    discovery_limit = args.max_images_per_person if args.first_photo_only else 0
    images = discover_images(
        roots,
        allowed_ids,
        discovery_limit,
        source_aliases,
        global_aliases,
    )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-count must be >= 1 and --shard-index must be in range.")
    if args.first_photo_only:
        if args.limit > 0:
            images = images[: args.limit]
        images = images[args.shard_index :: args.shard_count]
        person_count = len({item[1] for item in images})
    else:
        person_groups = group_images_by_person(images)
        if args.limit > 0:
            person_groups = person_groups[: args.limit]
        person_groups = person_groups[args.shard_index :: args.shard_count]
        images = [item for _, candidates in person_groups for item in candidates]
        person_count = len(person_groups)

    print(f"Study 2 IDs loaded: {len(allowed_ids) if allowed_ids else 'filter disabled'}")
    print(f"Input roots: {len(roots)}")
    print(f"Study 2 people selected: {person_count}")
    print(f"Candidate images selected: {len(images)}")
    print("Output directory configured.")
    if args.dry_run:
        for index, (source, person_id, _, relative) in enumerate(images[:10], start=1):
            detail = (
                f"{person_id} | {relative.as_posix()}"
                if args.show_identifiers
                else f"candidate_{index:04d}"
            )
            print(f"  {source} | {detail}")
        return 0
    if not images:
        print("No matching images were found.", file=sys.stderr)
        return 2

    from face_utils import (
        FacePlusPlusError,
        PersonSegmenter,
        get_faceplusplus_outputs,
        load_api_keys,
        preprocess_portrait,
        scale_landmarks,
    )

    credentials = load_api_keys(args.env_file)
    person_segmenter = PersonSegmenter(args.segmentation_model)
    manifest_name = (
        "processing_manifest.csv"
        if args.shard_count == 1
        else f"processing_manifest_shard_{args.shard_index}.csv"
    )
    manifest = args.output_dir / manifest_name
    completed: set[str] = set()
    completed_people: set[str] = set()
    terminal_failures: set[str] = set()
    if not args.no_resume:
        for prior_manifest in args.output_dir.glob("processing_manifest*.csv"):
            paths, people, terminal = load_manifest_state(prior_manifest)
            completed.update(paths)
            completed_people.update(people)
            terminal_failures.update(terminal)
    ok_count = 0
    failed_count = 0
    skipped_count = 0

    for index, (source, person_id, path, relative) in enumerate(images, start=1):
        portable_input = manifest_input_path(source, relative)
        input_key = portable_input.casefold()
        if person_id in completed_people or input_key in completed:
            skipped_count += 1
            continue
        if not args.first_photo_only and input_key in terminal_failures:
            skipped_count += 1
            continue
        raw_path, background_path, framed_path, cropped_path, gray_path = output_paths(
            args.output_dir, person_id, relative
        )
        row: dict[str, object] = {
            "input_path": portable_input,
            "source": source,
            "person_id": person_id,
            "output_raw": manifest_output_path(args.output_dir, raw_path),
            "output_background_removed": manifest_output_path(
                args.output_dir, background_path
            ),
            "output_framed": manifest_output_path(args.output_dir, framed_path),
            "output_grayscale": manifest_output_path(args.output_dir, gray_path),
            "output_cropped": manifest_output_path(args.output_dir, cropped_path),
            "status": "failed",
            "quality_score": "",
            "credential": "",
            "error": "",
            "rectangle_json": "",
            "landmarks_json": "",
            "attributes_json": "",
        }
        try:
            image = read_image(path)
            api_image = prepare_faceplusplus_image(image)
            result = get_faceplusplus_outputs(
                api_image,
                credentials,
                verify_score=args.verify_score,
                max_faces=args.max_faces,
                timeout=args.timeout,
            )
            source_landmarks = scale_landmarks(
                result["landmarks"],
                image.shape[1] / api_image.shape[1],
                image.shape[0] / api_image.shape[0],
            )
            processed = preprocess_portrait(
                image,
                source_landmarks,
                person_segmenter,
                grayscale_background=args.grayscale_background,
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, raw_path)
            write_image(background_path, processed.background_removed)
            write_image(framed_path, processed.framed)
            write_image(cropped_path, processed.cropped)
            write_image(gray_path, processed.grayscale)
            row.update(
                {
                    "status": "ok",
                    "quality_score": result["score"],
                    "segmentation_foreground_fraction": round(
                        processed.foreground_fraction, 6
                    ),
                    "credential": result["credential"],
                    "rectangle_json": json.dumps(result["rectangle"], ensure_ascii=False),
                    "landmarks_json": json.dumps(result["landmarks"], ensure_ascii=False),
                    "attributes_json": json.dumps(result["attributes"], ensure_ascii=False),
                }
            )
            ok_count += 1
            completed.add(input_key)
            completed_people.add(person_id)
        except (FacePlusPlusError, OSError, ValueError) as exc:
            row["error"] = privacy_safe_error(exc)
            if is_terminal_photo_error(str(row["error"])):
                terminal_failures.add(input_key)
            failed_count += 1
        append_manifest(manifest, row)
        detail = f": {person_id} / {path.name}" if args.show_identifiers else ""
        print(f"[{index}/{len(images)}] {row['status']}{detail}", flush=True)
        if args.request_delay > 0:
            time.sleep(args.request_delay)

    person_segmenter.close()

    print(
        f"Finished. ok={ok_count}, failed={failed_count}, resumed/skipped={skipped_count}; "
        f"manifest={manifest.name}"
    )
    return 0 if not args.first_photo_only or failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
