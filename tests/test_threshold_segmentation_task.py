from pathlib import Path

import numpy as np
import pytest
from ngio import open_ome_zarr_container
from skimage.draw import disk

from fcf_ome_zarr.converters.fcf_bd_converter import fcf_bd_converter
from fcf_ome_zarr.threshold_segmentation_task import (
    DEFAULT_CHANNEL,
    FOV_ROI_TABLE,
    segment_cell,
    threshold_segmentation_task,
)

LABEL_NAME = "LightLoss_Imaging_thresholded"
TABLE_NAME = f"{LABEL_NAME}_to_event"

# Synthetic FOVs mimic the LightLoss channel of the example data:
# float32, bright background around 0.15, cells darker
FOV_SHAPE = (73, 104)
BACKGROUND = 0.15
CELL = 0.1


@pytest.fixture(params=[False, True], ids=["yx", "z_singleton"])
def zarr_url(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    bd_s8_tiff_dir: Path,
    bd_s8_fcs_path: Path,
):
    """OME-Zarr produced by the BD converter from the example data."""
    result = fcf_bd_converter(
        tiff_folder_path=str(bd_s8_tiff_dir),
        zarr_dir=str(tmp_path),
        fcs_file_path=str(bd_s8_fcs_path),
        fake_a_plate=False,
        add_z_singleton=request.param,
    )
    assert result is not None
    return result["image_list_updates"][0]["zarr_url"]


@pytest.mark.parametrize(
    "params", [{}, {"sigma": (2.0, 1.5), "max_size": 32}], ids=["default", "custom"]
)
def test_roundtrip_per_fov(zarr_url: str, params: dict):
    """The label written for each FOV is segment_cell on that FOV, up to an offset."""
    threshold_segmentation_task(zarr_url=zarr_url, **params)

    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    label = ome_zarr.get_label(LABEL_NAME)

    seen: set[int] = set()
    for roi in ome_zarr.get_roi_table(FOV_ROI_TABLE).rois():
        fov = image.get_roi_as_numpy(
            roi, channel_selection=DEFAULT_CHANNEL, axes_order="yx"
        )
        expected = segment_cell(fov, **params)
        written = label.get_roi_as_numpy(roi, axes_order="yx")

        assert expected.max() > 0, f"No cell found in FOV {roi.name}"
        # Same background, and each object shifted by one constant offset
        np.testing.assert_array_equal(written == 0, expected == 0)
        offsets = np.unique(written[expected > 0] - expected[expected > 0])
        assert offsets.size == 1

        # Label ids are not shared between FOVs
        fov_labels = set(np.unique(written).tolist()) - {0}
        assert not fov_labels & seen
        seen |= fov_labels


def test_label_to_event_table(zarr_url: str):
    """Every label is mapped to the event of the FOV it was segmented in."""
    threshold_segmentation_task(zarr_url=zarr_url)

    ome_zarr = open_ome_zarr_container(zarr_url)
    label = ome_zarr.get_label(LABEL_NAME)
    table = ome_zarr.get_feature_table(TABLE_NAME)
    assert table.reference_label == LABEL_NAME
    df = table.dataframe
    assert df.index.name == "label"

    expected: dict[int, int] = {}
    for roi in ome_zarr.get_roi_table(FOV_ROI_TABLE).rois():
        fov_labels = set(np.unique(label.get_roi_as_numpy(roi)).tolist()) - {0}
        expected |= dict.fromkeys(fov_labels, int(roi.get_name()))
    assert df.index.is_unique
    mapping = dict(
        zip(df.index.astype(int), df["event_index"].astype(int), strict=True)
    )
    assert mapping == expected

    # Event indices link to the FCS events
    events = ome_zarr.get_feature_table("fcs_events").dataframe
    assert set(mapping.values()) <= set(events.index.astype(int))


def test_custom_label_name(zarr_url: str):
    threshold_segmentation_task(zarr_url=zarr_url, label_name="cells")

    tables = open_ome_zarr_container(zarr_url).list_tables()
    assert "cells_to_event" in tables


def test_no_overwrite(zarr_url: str):
    threshold_segmentation_task(zarr_url=zarr_url)
    with pytest.raises(FileExistsError):
        threshold_segmentation_task(zarr_url=zarr_url, overwrite=False)


def _fov() -> np.ndarray:
    """Noisy background with no cell."""
    rng = np.random.default_rng(0)
    return (BACKGROUND + rng.normal(0, 0.002, FOV_SHAPE)).astype(np.float32)


def _disk_mask(center: tuple[int, int], radius: float) -> np.ndarray:
    mask = np.zeros(FOV_SHAPE, dtype=bool)
    mask[disk(center, radius, shape=FOV_SHAPE)] = True
    return mask


def test_segment_cell_single_dark_cell():
    img = _fov()
    img[_disk_mask((36, 52), 12)] = CELL

    label_img = segment_cell(img)

    assert label_img.shape == FOV_SHAPE
    assert set(np.unique(label_img).tolist()) == {0, 1}
    # The object covers the cell and doesn't spread far beyond it
    assert label_img[_disk_mask((36, 52), 10)].all()
    assert not label_img[~_disk_mask((36, 52), 16)].any()


def test_segment_cell_keeps_separate_objects():
    img = _fov()
    img[_disk_mask((36, 25), 10)] = CELL
    img[_disk_mask((36, 80), 10)] = CELL

    label_img = segment_cell(img)

    assert set(np.unique(label_img).tolist()) == {0, 1, 2}
    assert label_img[36, 25] != label_img[36, 80]


def test_segment_cell_fills_holes():
    img = _fov()
    img[_disk_mask((36, 52), 12)] = CELL
    img[_disk_mask((36, 52), 5)] = BACKGROUND

    label_img = segment_cell(img)

    assert set(np.unique(label_img).tolist()) == {0, 1}
    assert label_img[_disk_mask((36, 52), 5)].all()
