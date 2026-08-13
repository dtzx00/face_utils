# face_utils

This branch contains the reusable Study 2 portrait-preprocessing pipeline. It
ends after grayscale conversion and deliberately excludes person-ID merging,
classification, matching, Face++ re-analysis, DeepFace, VGGFace, fWHR, and
statistical analysis.

## Pipeline

1. Discover candidate portraits in one or more caller-supplied directories.
2. Optionally restrict candidates to IDs in a caller-supplied CSV or XLSX file.
3. Try candidate portraits for each ID in discovery order until one succeeds.
4. Send a JPEG-encoded copy of the unmodified portrait to Face++ and request
   attributes plus 106-point landmarks.
5. Record the 0–5 usability score without filtering by default.
6. Segment the person on the complete raw portrait and remove the background.
7. Draw the blue crop frame and purple face contour on a diagnostic copy.
8. Rotate around the eye midpoint until the eye line is horizontal.
9. Build a dynamic square from the aligned face contour and resize it to
   224×224 pixels.
10. Composite the transparent crop onto neutral gray and save an 8-bit,
    single-channel grayscale PNG.

The workflow writes `raw`, `background_removed`, `framed`, `cropped`, and
`grayscale` outputs plus a resumable manifest. Source images are never changed.

## Setup

```powershell
git clone <repository-url>
cd face_utils
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `FPP_KEY` and `FPP_SECRET` in the local `.env`. The file is ignored by Git
and must never be committed.

If credentials already exist as `API_KEY` and `API_SECRET` assignments in a
private Python file, migrate them without displaying their values:

```powershell
python migrate_fpp_credentials.py --source .\private\legacy_config.py
```

## Discover inputs safely

Supply either `--study2-root` for a directory containing `Raw - *` folders, or
one or more `--input-root` arguments. Supply `--ids-file` unless filtering is
explicitly disabled.

```powershell
python process_study2.py `
  --input-root .\data\portraits `
  --ids-file .\data\participants.xlsx `
  --dry-run
```

Dry-run mode does not load credentials or upload images.

## Pilot and full run

Face images are sent to Face++ by these commands:

```powershell
python process_study2.py `
  --input-root .\data\portraits `
  --ids-file .\data\participants.xlsx `
  --limit 10
```

After reviewing the pilot outputs, omit `--limit` for the full run:

```powershell
python process_study2.py `
  --input-root .\data\portraits `
  --ids-file .\data\participants.xlsx
```

The default `--verify-score 0` records the original 0–5 score but does not
discard low-scoring portraits. Pass `--verify-score 5` to reproduce the
original strict screening rule.

No-face, multi-face, and decode failures cause the next portrait for the same
ID to be tried. Processing stops for that ID after its first success. Use
`--first-photo-only` to disable this fallback behavior.

Trailing `STAR`, `_STAR`, `*`, `_`, and `-` markers are identity-significant.
Each marker becomes one filesystem-safe trailing underscore, so a base ID and
its marked variant remain distinct.

Successful IDs are skipped when a run resumes. Use `--no-resume` only when a
complete rerun is intended.

## Reprocess saved landmarks

Existing successful manifests can be reprocessed with the new segmentation and
crop stages without another paid Face++ request:

```powershell
python reprocess_background_from_manifest.py `
  --source-output-dir .\previous_output `
  --output-dir .\reprocessed_output
```

## Outputs

```text
study2_preprocessed_output/
  raw/
  background_removed/
  framed/
  cropped/
  grayscale/
  processing_manifest.csv
```

All image folders are flat. New manifests identify source files with stable
hashed tokens and store output locations as portable relative paths rather than
workstation paths. They also retain segmentation coverage, the Face++ rectangle,
attributes, and landmarks so preprocessing can be reproduced without another
API call.

## Privacy

The repository contains no credentials, participant records, workstation
paths, institution-specific locations, or real participant-ID examples. Input
directories and ID files are always provided at runtime. Source labels written
to new manifests are generic aliases such as `input_01`.

Generated images and manifests still contain the caller's participant IDs by
design. Treat those outputs as research data and do not commit or publish them.
The default output directory and `.env` are excluded by `.gitignore`.

## Repository layout

```text
face_utils/
  face_utils/
    __init__.py
    utils.py
  models/
    selfie_segmenter.tflite
  tests/
    test_smoke.py
  process_study2.py
  reprocess_background_from_manifest.py
  migrate_fpp_credentials.py
  requirements.txt
  setup.py
```

## Important changes from the original repository

- API and network errors are recorded instead of silently becoming zeroes.
- One credential pair is used: `FPP_KEY` and `FPP_SECRET`.
- Cropping levels the eyes without forcing them to fixed template coordinates,
  then uses the complete Face++ contour and dynamic margins.
- Background removal is estimated on the complete raw portrait before framing,
  rotation, and cropping.
- Final grayscale images use a uniform light-gray background.
- `to_grayscale()` normalizes 8-bit input before clipping.
- Unicode paths are read and written through OpenCV-safe byte methods.
