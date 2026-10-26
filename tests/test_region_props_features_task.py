from pathlib import Path

import numpy as np
import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.converters.fcf_s8_converter import fcf_s8_converter
from fcf_ome_zarr.region_props_features_task import (
    INTENSITY_PROPERTIES,
    region_props_features_task,
)
from fcf_ome_zarr.threshold_segmentation_task import threshold_segmentation_task

LABEL_NAME = "LightLoss_Imaging_thresholded"
TABLE_NAME = "region_props_features"


@pytest.fixture(params=[False, True], ids=["yx", "z_singleton"])
def zarr_url(request: pytest.FixtureRequest, tmp_path: Path, example_data_dir: Path):
    """OME-Zarr from the S8 converter, segmented by the threshold segmentation."""
    result = fcf_s8_converter(
        tiff_folder_path=str(example_data_dir),
        zarr_dir=str(tmp_path),
        fake_a_plate=False,
        add_z_singleton=request.param,
    )
    assert result is not None
    zarr_url = result["image_list_updates"][0]["zarr_url"]
    threshold_segmentation_task(zarr_url=zarr_url)
    return zarr_url


def test_columns_named_by_channel(zarr_url: str, example_channel_labels: tuple):
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)

    df = open_ome_zarr_container(zarr_url).get_feature_table(TABLE_NAME).dataframe
    assert df.index.name == "label"
    expected = {
        f"{prop}-{channel}"
        for prop in INTENSITY_PROPERTIES
        for channel in example_channel_labels
    }
    assert expected <= set(df.columns)
    # No leftover columns suffixed with the channel index
    assert not [col for col in df.columns if col.rsplit("-", 1)[-1].isdigit()]


def test_features_match_image(zarr_url: str, example_channel_labels: tuple):
    """One row per object, with area and intensities computed on the image."""
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)

    ome_zarr = open_ome_zarr_container(zarr_url)
    df = ome_zarr.get_feature_table(TABLE_NAME).dataframe
    image = ome_zarr.get_image().get_as_numpy(axes_order="cyx")
    label = ome_zarr.get_label(LABEL_NAME).get_as_numpy(axes_order="yx")

    label_ids = sorted(set(np.unique(label).tolist()) - {0})
    assert len(label_ids) > 0
    assert sorted(df.index.astype(int).tolist()) == label_ids

    for label_id in label_ids:
        mask = label == label_id
        row = df.loc[df.index.astype(int) == label_id].iloc[0]
        assert row["area"] == mask.sum()
        for channel_img, channel in zip(image, example_channel_labels, strict=True):
            pixels = channel_img[mask].astype(np.float64)
            assert row[f"mean_intensity-{channel}"] == pytest.approx(
                pixels.mean(), rel=1e-5, abs=1e-9
            )
            assert row[f"max_intensity-{channel}"] == pytest.approx(
                pixels.max(), rel=1e-5, abs=1e-9
            )
            assert row[f"min_intensity-{channel}"] == pytest.approx(
                pixels.min(), rel=1e-5, abs=1e-9
            )


def test_no_overwrite(zarr_url: str):
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)
    with pytest.raises(FileExistsError):
        region_props_features_task(
            zarr_url=zarr_url, label_image_name=LABEL_NAME, overwrite=False
        )
