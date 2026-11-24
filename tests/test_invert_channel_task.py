from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.constants import (
    FCS_METADATA_FILE,
    FOV_ROI_TABLE,
    UINT16_MAX_INTENSITY,
)
from fcf_ome_zarr.invert_channel_task import invert_channel_task

CHANNEL = "LightLoss_Imaging"


@pytest.fixture(params=[False, True], ids=["float32", "uint16"])
def zarr_url(
    request: pytest.FixtureRequest,
    convert_bd_s8: Callable[..., str],
    bd_s8_fcs_path: Path,
) -> str:
    """OME-Zarr produced by the BD converter from the example data."""
    return convert_bd_s8(
        fcs_file_path=str(bd_s8_fcs_path), convert_to_uint16=request.param
    )


def _channel(zarr_url: str, label: str) -> np.ndarray:
    image = open_ome_zarr_container(zarr_url).get_image()
    return image.get_as_numpy(channel_selection=label, axes_order="yx")


def test_inverting_twice_gives_original(zarr_url: str):
    original = _channel(zarr_url, CHANNEL)

    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)
    invert_channel_task(zarr_url=zarr_url, channel_label=f"{CHANNEL}_inv")

    # Exact for uint16; float32 rounding of 8 - x is about 1e-7
    np.testing.assert_allclose(
        _channel(zarr_url, f"{CHANNEL}_inv_inv"), original, rtol=0, atol=1e-6
    )


def test_padding_stays_zero(zarr_url: str):
    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)

    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    in_fov = np.zeros(image.shape[-2:], dtype=bool)
    for roi in ome_zarr.get_roi_table(FOV_ROI_TABLE).rois():
        s = roi.to_slicing_dict(image.pixel_size)
        y = slice(round(s["y"].start), round(s["y"].stop))
        x = slice(round(s["x"].start), round(s["x"].stop))
        in_fov[y, x] = True
    # The example FOVs are smaller than the tiles, so the mosaic has padding
    assert not in_fov.all()

    inverted = _channel(zarr_url, f"{CHANNEL}_inv")
    assert (inverted[~in_fov] == 0).all()


def test_metadata_and_tables_kept(zarr_url: str):
    before = open_ome_zarr_container(zarr_url)
    channels = before.get_image().channels_meta.channels
    tables = before.list_tables()

    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)

    after = open_ome_zarr_container(zarr_url)
    *originals, inverted = after.get_image().channels_meta.channels
    assert originals == channels
    assert inverted.label == f"{CHANNEL}_inv"
    # The inverted window mirrors the source window: [c - end, c - start]
    c = 65535 if after.get_image().dtype == "uint16" else UINT16_MAX_INTENSITY
    source = next(ch for ch in channels if ch.label == CHANNEL).channel_visualisation
    window = (inverted.channel_visualisation.start, inverted.channel_visualisation.end)
    assert window == pytest.approx((c - source.end, c - source.start))
    assert after.list_tables() == tables
    assert (Path(zarr_url) / FCS_METADATA_FILE).exists()


def test_unknown_label_raises(zarr_url: str):
    with pytest.raises(ValueError, match="not found"):
        invert_channel_task(zarr_url=zarr_url, channel_label="unknown")


def test_existing_inverted_label_raises(zarr_url: str):
    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)
    with pytest.raises(ValueError, match="already exists"):
        invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)
