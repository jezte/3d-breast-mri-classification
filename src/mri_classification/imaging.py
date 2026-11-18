
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage import io

SERIES_NAMES = {"pre", "post_1", "post_2"}


def central_slice_number(annotation: pd.Series) -> int:
    """Return the integer midpoint of the lesion's annotated slice range."""
    return (int(annotation["Start Slice"]) + int(annotation["End Slice"])) // 2


def create_masks(
    image_shape: tuple[int, ...],
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a rectangular lesion mask and its horizontal mirror.

    The mirrored mask represents negative-label.
    Validate this assumption before treating the mirrored region as a true non-lesion.
    """
    lesion_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    lesion_mask[start_row:end_row, start_col:end_col] = 255
    mirrored_mask = np.flip(lesion_mask, axis=1).copy()
    return lesion_mask, mirrored_mask


def prepare_extracted_slices(
    annotation_boxes: pd.DataFrame,
    filepath_mapping: pd.DataFrame,
    dicom_root: str | Path,
    output_root: str | Path,
) -> int:
    """Save annotated DICOM slices and their lesion/mirrored masks as PNG files."""
    dicom_root = Path(dicom_root)
    output_root = Path(output_root)

    output_count = 0

    for row in filepath_mapping.itertuples(index=False):
        # Parse the original path to identify patient, series, and slice.
        original = Path(str(row.original_path_and_filename).replace("\\", "/"))
        parts = original.parts

        patient_part = next((part for part in parts if part.startswith("Breast_MRI_")), None)
        series_name = next((part for part in parts if part in SERIES_NAMES), None)

        if patient_part is None or series_name is None:
            continue

        try:
            patient_num = int(patient_part[-3:])
            slice_num = int(original.stem.rsplit("_", 1)[-1])

            annotation = annotation_boxes.iloc[patient_num - 1]

            first_slice = int(annotation["Start Slice"])
            last_slice = int(annotation["End Slice"])

        except (ValueError, IndexError, TypeError):
            continue

        # Process only slices that contain lesion.
        if not first_slice <= slice_num <= last_slice:
            continue

        # Resolve the actual DICOM file using the classic path.
        classic_path = Path(str(row.classic_path).replace("\\", "/"))

        if classic_path.parts[0] == dicom_root.name:
            classic_path = Path(*classic_path.parts[1:])

        dicom_path = dicom_root / classic_path

        if not dicom_path.is_file():
            raise FileNotFoundError(
                f"Mapped DICOM file not found: {dicom_path}"
            )

        dataset = pydicom.dcmread(dicom_path)
        image = dataset.pixel_array

        # Create the lesion and mirrored masks.
        lesion_mask, mirrored_mask = create_masks(
            image.shape,
            int(annotation["Start Row"]),
            int(annotation["End Row"]),
            int(annotation["Start Column"]),
            int(annotation["End Column"]),
        )

        # Preserve the descriptive patient/series structure in the outputs.
        relative_png = Path(
            patient_part,
            series_name,
            original.name,
        ).with_suffix(".png")

        outputs = {
            "DICOM_Images": image,
            "lesion_masks": lesion_mask,
            "non_lesion_masks": mirrored_mask,
        }

        for folder, pixels in outputs.items():
            target = output_root / folder / relative_png
            target.parent.mkdir(parents=True, exist_ok=True)

            io.imsave(
                str(target),
                np.asarray(pixels),
                check_contrast=False,
            )

        output_count += 1

    return output_count
