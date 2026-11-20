from pathlib import Path

import numpy as np
import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.converters.fcf_bd_converter import fcf_bd_converter
from fcf_ome_zarr.label_to_event_task import label_to_event_task


@pytest.fixture
def zarr_url(tmp_path: Path, bd_s8_tiff_dir: Path, bd_s8_fcs_path: Path) -> str:
    """OME-Zarr produced by the BD converter from the example data."""
    result = fcf_bd_converter(
        tiff_folder_path=str(bd_s8_tiff_dir),
        zarr_dir=str(tmp_path),
        fcs_file_path=str(bd_s8_fcs_path),
        fake_a_plate=False,
    )
    assert result is not None
    return result["image_list_updates"][0]["zarr_url"]


def _write_label(zarr_url: str, label_img: np.ndarray) -> None:
    label = open_ome_zarr_container(zarr_url).derive_label(name="cells")
    label.set_array(label_img)
    label.consolidate()


def test_label_to_event_table(zarr_url: str):
    """Each label is mapped to the event of the FOV it lies in."""
    # FOV (y, x) extents: 0 -> [0:41, 0:104], 1 -> [0:73, 104:208],
    # 2 -> [73:142, 0:104], 3 -> [73:146, 104:208]
    label_img = np.zeros((146, 208), dtype=np.uint32)
    label_img[10:20, 10:20] = 5
    label_img[10:20, 150:160] = 3
    label_img[100:110, 150:160] = 2
    label_img[120:125, 120:125] = 7
    _write_label(zarr_url, label_img)

    label_to_event_task(zarr_url=zarr_url, label_name="cells")

    table = open_ome_zarr_container(zarr_url).get_feature_table("cells_to_event")
    assert table.reference_label == "cells"
    df = table.dataframe
    assert df.index.name == "label"
    mapping = dict(zip(df.index, df["event_index"], strict=True))
    assert mapping == {5: 0, 3: 1, 2: 3, 7: 3}


def test_label_spanning_fovs(zarr_url: str):
    label_img = np.zeros((146, 208), dtype=np.uint32)
    # One label across the boundary between the two top FOVs (tile size 73 x 104)
    label_img[30:40, 100:110] = 1
    _write_label(zarr_url, label_img)

    with pytest.raises(ValueError, match="more than one FOV"):
        label_to_event_task(zarr_url=zarr_url, label_name="cells")


def test_label_outside_fovs(zarr_url: str):
    label_img = np.zeros((146, 208), dtype=np.uint32)
    # FOV 0 is 41 px high, so its tile is padded below
    label_img[30:50, 10:20] = 1
    _write_label(zarr_url, label_img)

    with pytest.raises(ValueError, match="outside the FOVs"):
        label_to_event_task(zarr_url=zarr_url, label_name="cells")


def test_no_overwrite(zarr_url: str):
    _write_label(zarr_url, np.zeros((146, 208), dtype=np.uint32))
    label_to_event_task(zarr_url=zarr_url, label_name="cells")
    with pytest.raises(FileExistsError):
        label_to_event_task(zarr_url=zarr_url, label_name="cells", overwrite=False)
