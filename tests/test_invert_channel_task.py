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
from fcf_ome_zarr.converters.utils import Shards
from fcf_ome_zarr.invert_channel_task import invert, invert_channel_task

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


def _background(zarr_url: str) -> float:
    """Median of the per-FOV medians of the source channel."""
    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    rois = ome_zarr.get_roi_table(FOV_ROI_TABLE).rois()
    return float(
        np.median(
            [
                np.median(image.get_roi_as_numpy(r, channel_selection=CHANNEL))
                for r in rois
            ]
        )
    )


def test_invert():
    floats = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    np.testing.assert_array_equal(invert(floats, 1.0), [0.5, 0.0, 0.0])
    assert invert(floats, 1.0).dtype == np.float32

    uints = np.array([100, 200, 300], dtype=np.uint16)
    np.testing.assert_array_equal(invert(uints, 250.4), [150, 50, 0])
    assert invert(uints, 250.4).dtype == np.uint16


def test_float32_and_uint16_match(convert_bd_s8: Callable[..., str]):
    float_url = convert_bd_s8(zarr_name="float32", convert_to_uint16=False)
    uint_url = convert_bd_s8(zarr_name="uint16", convert_to_uint16=True)
    invert_channel_task(zarr_url=float_url, channel_label=CHANNEL)
    invert_channel_task(zarr_url=uint_url, channel_label=CHANNEL)

    inverted = _channel(float_url, f"{CHANNEL}_inv")
    assert inverted.max() > 0
    # uint16 = round(float * scale); c and x each round by up to 0.5
    scale = np.iinfo(np.uint16).max / UINT16_MAX_INTENSITY
    np.testing.assert_allclose(
        _channel(uint_url, f"{CHANNEL}_inv") / scale, inverted, rtol=0, atol=2 / scale
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
    # The window is inverted like the data: max(0, c - x)
    c = _background(zarr_url)
    source = next(ch for ch in channels if ch.label == CHANNEL).channel_visualisation
    window = (inverted.channel_visualisation.start, inverted.channel_visualisation.end)
    assert window == pytest.approx((max(0, c - source.end), max(0, c - source.start)))
    assert after.list_tables() == tables
    assert (Path(zarr_url) / FCS_METADATA_FILE).exists()


def test_unknown_label_raises(zarr_url: str):
    with pytest.raises(ValueError, match="not found"):
        invert_channel_task(zarr_url=zarr_url, channel_label="unknown")


def test_existing_inverted_label_raises(zarr_url: str):
    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)
    with pytest.raises(ValueError, match="already exists"):
        invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)


def test_sharded_image(convert_bd_s8: Callable[..., str]):
    zarr_url = convert_bd_s8(storage=Shards(target_shard_size=150))
    invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)

    image = open_ome_zarr_container(zarr_url).get_image()
    shards = image.zarr_array.shards
    assert shards is not None
    # Chunks and shards hold all channels, including the new one
    assert image.chunks[0] == shards[0] == len(image.channel_labels)


def test_unsupported_dtype_raises(convert_bd_s8: Callable[..., str], tmp_path: Path):
    int_url = str(tmp_path / "int16.ome.zarr")
    open_ome_zarr_container(convert_bd_s8()).derive_image(store=int_url, dtype="int16")
    with pytest.raises(TypeError, match="Cannot invert"):
        invert_channel_task(zarr_url=int_url, channel_label=CHANNEL)


def test_leftover_derived_image_raises(convert_bd_s8: Callable[..., str]):
    zarr_url = convert_bd_s8()
    Path(f"{zarr_url}_inverted").mkdir()
    with pytest.raises(FileExistsError, match="failed run"):
        invert_channel_task(zarr_url=zarr_url, channel_label=CHANNEL)
