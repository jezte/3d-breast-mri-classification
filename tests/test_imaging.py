from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mri_classification import imaging


def test_central_slice_number():
    annotation = pd.Series({
        "Start Slice": 10,
        "End Slice": 18,
    })

    assert imaging.central_slice_number(annotation) == 14


def test_create_masks():
    lesion_mask, mirrored_mask = imaging.create_masks(
        image_shape=(10, 12),
        start_row=2,
        end_row=5,
        start_col=3,
        end_col=6,
    )

    assert lesion_mask.shape == (10, 12)
    assert mirrored_mask.shape == (10, 12)

    assert np.all(lesion_mask[2:5, 3:6] == 255)
    assert np.count_nonzero(lesion_mask) == 9

    assert np.array_equal(mirrored_mask, np.fliplr(lesion_mask))


def test_prepare_extracted_slices(tmp_path, monkeypatch):
    dicom_root = tmp_path / "Duke-Breast-Cancer-MRI"
    output_root = tmp_path / "extracted"

    classic_path = (
        "Duke-Breast-Cancer-MRI/Breast_MRI_001/"
        "1.3.6.1.4.1.14519.5.2.1.186051521067863971269584893740842397538/"
        "1.3.6.1.4.1.14519.5.2.1.15147203499230039585785258602759759529/"
        "1-014.dcm"
    )

    dicom_file = dicom_root / Path(classic_path).relative_to(
        "Duke-Breast-Cancer-MRI"
    )
    dicom_file.parent.mkdir(parents=True)
    dicom_file.touch()

    annotation_boxes = pd.DataFrame([
        {
            "Patient ID": 1,
            "Start Slice": 10,
            "End Slice": 18,
            "Start Row": 2,
            "End Row": 5,
            "Start Column": 3,
            "End Column": 6,
        }
    ])

    filepath_mapping = pd.DataFrame([
        {
            "original_path_and_filename":
                "DICOM_Images/Breast_MRI_001/pre/Breast_MRI_001_pre_014.dcm",
            "classic_path": classic_path,
        },
        {
            "original_path_and_filename":
                "DICOM_Images/Breast_MRI_001/pre/Breast_MRI_001_pre_020.dcm",
            "classic_path": classic_path.replace("014", "020"),
        },
    ])

    image = np.arange(100).reshape(10, 10)

    def fake_dcmread(path):
        return SimpleNamespace(pixel_array=image)

    monkeypatch.setattr(
        imaging.pydicom,
        "dcmread",
        fake_dcmread,
    )

    saved_images = {}

    def fake_imsave(path, pixels, check_contrast=False):
        saved_images[path] = pixels.copy()

    monkeypatch.setattr(imaging.io, "imsave", fake_imsave)

    output_count = imaging.prepare_extracted_slices(
        annotation_boxes,
        filepath_mapping,
        dicom_root,
        output_root,
    )

    assert output_count == 1
    assert len(saved_images) == 3
