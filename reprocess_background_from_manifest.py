"""Rebuild Study 2 outputs with background removal without calling Face++."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from face_utils import (
    FacePlusPlusError,
    PersonSegmenter,
    preprocess_portrait,
    scale_landmarks,
)
from process_study2 import (
    DEFAULT_SEGMENTATION_MODEL,
    append_manifest,
    load_manifest_state,
    manifest_input_path,
    manifest_output_path,
    normalize_id,
    output_paths,
    prepare_faceplusplus_image,
    privacy_safe_error,
    read_image,
    write_image,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use saved Face++ landmarks to run raw background removal, frame, "
            "crop, and grayscale without another API request."
        )
    )
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segmentation-model", type=Path, default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument(
        "--grayscale-background", type=int, choices=range(0, 256), default=238
    )
    parser.add_argument("--person-id", action="append", help="Process only this ID; repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N people; 0 means all.")
    parser.add_argument("--show-identifiers", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def successful_rows(source_output: Path) -> list[dict[str, str]]:
    """Return one usable successful row per normalized person ID."""
    by_person: dict[str, dict[str, str]] = {}
    for manifest in sorted(source_output.glob("processing_manifest*.csv")):
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                person_id = normalize_id(row.get("person_id", ""))
                stored_raw = Path(row.get("output_raw", ""))
                candidates = (
                    [stored_raw]
                    if stored_raw.is_absolute()
                    else [source_output / stored_raw, stored_raw]
                )
                raw = next((path for path in candidates if path.exists()), candidates[0])
                if (
                    row.get("status") == "ok"
                    and person_id
                    and row.get("landmarks_json")
                    and raw.exists()
                ):
                    row["_resolved_output_raw"] = str(raw)
                    by_person[person_id] = row
    return list(by_person.values())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args()
    rows = successful_rows(args.source_output_dir)
    if args.person_id:
        wanted = {normalize_id(value) for value in args.person_id}
        rows = [row for row in rows if normalize_id(row["person_id"]) in wanted]
    if args.limit > 0:
        rows = rows[: args.limit]
    manifest = args.output_dir / "processing_manifest.csv"
    completed_people: set[str] = set()
    if not args.no_resume:
        _, completed_people, _ = load_manifest_state(manifest)

    ok_count = failed_count = skipped_count = 0
    with PersonSegmenter(args.segmentation_model) as segmenter:
        for index, prior in enumerate(rows, start=1):
            person_id = normalize_id(prior["person_id"])
            if person_id in completed_people:
                skipped_count += 1
                continue
            source_raw = Path(prior["_resolved_output_raw"])
            raw_path, background_path, framed_path, cropped_path, gray_path = output_paths(
                args.output_dir, person_id, source_raw
            )
            row: dict[str, object] = {
                "input_path": manifest_input_path("source", Path(source_raw.name)),
                "source": "saved_manifest",
                "person_id": person_id,
                "output_raw": manifest_output_path(args.output_dir, raw_path),
                "output_background_removed": manifest_output_path(
                    args.output_dir, background_path
                ),
                "output_framed": manifest_output_path(args.output_dir, framed_path),
                "output_cropped": manifest_output_path(args.output_dir, cropped_path),
                "output_grayscale": manifest_output_path(args.output_dir, gray_path),
                "status": "failed",
                "quality_score": prior.get("quality_score", ""),
                "segmentation_foreground_fraction": "",
                "credential": "saved_landmarks_no_api_call",
                "error": "",
                "rectangle_json": prior.get("rectangle_json", ""),
                "landmarks_json": prior.get("landmarks_json", ""),
                "attributes_json": prior.get("attributes_json", ""),
            }
            try:
                image = read_image(source_raw)
                api_image = prepare_faceplusplus_image(image)
                landmarks = {
                    name: (float(point[0]), float(point[1]))
                    for name, point in json.loads(prior["landmarks_json"]).items()
                }
                source_landmarks = scale_landmarks(
                    landmarks,
                    image.shape[1] / api_image.shape[1],
                    image.shape[0] / api_image.shape[0],
                )
                processed = preprocess_portrait(
                    image,
                    source_landmarks,
                    segmenter,
                    grayscale_background=args.grayscale_background,
                )

                raw_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_raw, raw_path)
                write_image(background_path, processed.background_removed)
                write_image(framed_path, processed.framed)
                write_image(cropped_path, processed.cropped)
                write_image(gray_path, processed.grayscale)
                row["status"] = "ok"
                row["segmentation_foreground_fraction"] = round(
                    processed.foreground_fraction, 6
                )
                completed_people.add(person_id)
                ok_count += 1
            except (FacePlusPlusError, OSError, ValueError) as exc:
                row["error"] = privacy_safe_error(exc)
                failed_count += 1
            append_manifest(manifest, row)
            detail = (
                f": {person_id} / {source_raw.name}"
                if args.show_identifiers
                else ""
            )
            print(f"[{index}/{len(rows)}] {row['status']}{detail}", flush=True)

    print(
        f"Finished. ok={ok_count}, failed={failed_count}, "
        f"resumed/skipped={skipped_count}; manifest={manifest.name}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
