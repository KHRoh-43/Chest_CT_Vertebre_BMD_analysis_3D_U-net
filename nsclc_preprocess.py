from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm


DATA_ROOT = Path(r"C:\Users\comds\Desktop\졸업작품\dataset\NSCLC LUNG CANCER")
NIFTI_ROOT = DATA_ROOT / "nifti_output"
METADATA_PATH = DATA_ROOT / "metadata.csv"
OUTPUT_ROOT = DATA_ROOT / "nsclc_resampled_1x1x2"
OUTPUT_IMAGE_ROOT = OUTPUT_ROOT / "images"

TARGET_SPACING = (1.0, 1.0, 2.0)
MIN_CANONICAL_SLICES = 75
MAX_SLICE_MISMATCH_FOR_USE = 5


def make_new_affine(old_affine: np.ndarray, target_spacing: tuple[float, float, float]) -> np.ndarray:
    new_affine = old_affine.copy()
    for axis in range(3):
        direction = old_affine[:3, axis]
        norm = np.linalg.norm(direction)
        if norm > 0:
            new_affine[:3, axis] = direction / norm * target_spacing[axis]
    return new_affine


def resample_array(
    arr: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    order: int,
) -> np.ndarray:
    zoom_factors = [original_spacing[i] / target_spacing[i] for i in range(3)]
    return zoom(arr, zoom=zoom_factors, order=order)


def load_ct_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_PATH)
    ct = df[df["Modality"] == "CT"].copy()
    ct["Number of Images"] = pd.to_numeric(ct["Number of Images"], errors="coerce")
    return ct


def build_manifest() -> pd.DataFrame:
    ct = load_ct_metadata()
    ct_by_subject = ct.set_index("Subject ID", drop=False)
    rows = []

    for image_path in sorted(NIFTI_ROOT.glob("*.nii.gz")):
        file_name = image_path.name
        canonical = bool(re.match(r"^LUNG1-\d{3}\.nii\.gz$", file_name))
        subject_match = re.match(r"^(LUNG1-\d{3})", file_name)
        subject_id = subject_match.group(1) if subject_match else image_path.name.replace(".nii.gz", "")

        nii = nib.load(str(image_path))
        spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
        shape = tuple(int(v) for v in nii.shape[:3])
        meta = ct_by_subject.loc[subject_id] if subject_id in ct_by_subject.index else None
        metadata_images = int(meta["Number of Images"]) if meta is not None and pd.notna(meta["Number of Images"]) else None
        slice_mismatch = abs(shape[2] - metadata_images) if metadata_images is not None else None

        exclude_reason = ""
        if not canonical:
            exclude_reason = "noncanonical_extra_output"
        elif shape[2] < MIN_CANONICAL_SLICES:
            exclude_reason = "too_few_slices"
        elif slice_mismatch is not None and slice_mismatch > MAX_SLICE_MISMATCH_FOR_USE:
            exclude_reason = "metadata_slice_mismatch"

        rows.append(
            {
                "source": "NSCLC-Radiomics",
                "subject_id": subject_id,
                "file_name": file_name,
                "image_path": str(image_path),
                "canonical": canonical,
                "use_for_preprocessing": exclude_reason == "",
                "exclude_reason": exclude_reason,
                "shape_x": shape[0],
                "shape_y": shape[1],
                "shape_z": shape[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "physical_x_mm": shape[0] * spacing[0],
                "physical_y_mm": shape[1] * spacing[1],
                "physical_z_mm": shape[2] * spacing[2],
                "datatype": str(nii.header.get_data_dtype()),
                "metadata_number_of_images": metadata_images,
                "metadata_manufacturer": "" if meta is None else meta["Manufacturer"],
                "metadata_series_uid": "" if meta is None else meta["Series UID"],
                "metadata_file_location": "" if meta is None else meta["File Location"],
            }
        )

    manifest = pd.DataFrame(rows)
    return manifest.sort_values("file_name").reset_index(drop=True)


def summarize_intensity(image_paths: list[Path], max_cases: int) -> pd.DataFrame:
    rows = []
    for image_path in tqdm(image_paths[:max_cases], desc="Intensity check"):
        nii = nib.load(str(image_path))
        arr = np.asanyarray(nii.dataobj)
        percentiles = np.percentile(arr, [0, 0.1, 1, 50, 99, 99.5, 100])
        rows.append(
            {
                "file_name": image_path.name,
                "min": percentiles[0],
                "p0_1": percentiles[1],
                "p1": percentiles[2],
                "p50": percentiles[3],
                "p99": percentiles[4],
                "p99_5": percentiles[5],
                "max": percentiles[6],
            }
        )
    return pd.DataFrame(rows)


def resample_images(manifest: pd.DataFrame, limit: int | None, overwrite: bool) -> pd.DataFrame:
    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    target_df = manifest[manifest["use_for_preprocessing"]].copy()
    if limit is not None:
        target_df = target_df.head(limit)

    rows = []
    for row in tqdm(target_df.itertuples(index=False), total=len(target_df), desc="Resampling NSCLC"):
        image_path = Path(row.image_path)
        output_path = OUTPUT_IMAGE_ROOT / row.file_name
        if output_path.exists() and not overwrite:
            out = nib.load(str(output_path))
            rows.append(
                {
                    "file_name": row.file_name,
                    "status": "exists",
                    "input_shape": (row.shape_x, row.shape_y, row.shape_z),
                    "output_shape": out.shape,
                    "output_spacing": out.header.get_zooms()[:3],
                    "output_path": str(output_path),
                }
            )
            continue

        img_nii = nib.load(str(image_path))
        original_spacing = tuple(float(v) for v in img_nii.header.get_zooms()[:3])
        image_arr = np.asanyarray(img_nii.dataobj)
        image_resampled = resample_array(
            image_arr,
            original_spacing=original_spacing,
            target_spacing=TARGET_SPACING,
            order=1,
        ).astype(np.float32)

        new_affine = make_new_affine(img_nii.affine, TARGET_SPACING)
        out_img = nib.Nifti1Image(image_resampled, new_affine)
        out_img.header.set_zooms(TARGET_SPACING)
        nib.save(out_img, str(output_path))

        rows.append(
            {
                "file_name": row.file_name,
                "status": "saved",
                "input_shape": img_nii.shape,
                "input_spacing": original_spacing,
                "output_shape": image_resampled.shape,
                "output_spacing": TARGET_SPACING,
                "output_path": str(output_path),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--intensity-cases", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest_csv = OUTPUT_ROOT / "nsclc_manifest.csv"
    manifest_pkl = OUTPUT_ROOT / "nsclc_manifest.pkl"
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    manifest.to_pickle(manifest_pkl)

    usable = manifest[manifest["use_for_preprocessing"]]
    summary = {
        "n_nifti_files": int(len(manifest)),
        "n_usable_for_preprocessing": int(len(usable)),
        "n_excluded": int((~manifest["use_for_preprocessing"]).sum()),
        "target_spacing": TARGET_SPACING,
        "exclude_counts": manifest["exclude_reason"].replace("", "usable").value_counts().to_dict(),
        "shape_z": usable["shape_z"].describe().to_dict(),
        "spacing": usable[["spacing_x", "spacing_y", "spacing_z"]].describe().to_dict(),
    }
    (OUTPUT_ROOT / "nsclc_manifest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    intensity = summarize_intensity(
        [Path(p) for p in usable["image_path"].tolist()],
        max_cases=args.intensity_cases,
    )
    intensity.to_csv(OUTPUT_ROOT / "nsclc_intensity_sample.csv", index=False, encoding="utf-8-sig")

    if not args.manifest_only:
        log = resample_images(manifest, limit=args.limit, overwrite=args.overwrite)
        log.to_csv(OUTPUT_ROOT / "nsclc_resample_log.csv", index=False, encoding="utf-8-sig")
        log.to_pickle(OUTPUT_ROOT / "nsclc_resample_log.pkl")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
