import itertools
from collections.abc import Callable

import numpy as np
import pytest
from ngio import open_ome_zarr_container
from skimage import measure

from fcf_ome_zarr.region_props_features_task import (
    INTENSITY_PROPERTIES,
    region_props_features_task,
)
from fcf_ome_zarr.threshold_segmentation_task import threshold_segmentation_task

LABEL_NAME = "LightLoss_Imaging_thresholded"
TABLE_NAME = "region_props_features"


@pytest.fixture(params=[False, True], ids=["yx", "z_singleton"])
def zarr_url(request: pytest.FixtureRequest, convert_bd_s8: Callable[..., str]):
    """OME-Zarr from the BD converter, segmented by the threshold segmentation."""
    zarr_url = convert_bd_s8(add_z_singleton=request.param)
    threshold_segmentation_task(zarr_url=zarr_url)
    return zarr_url


def test_columns_named_by_channel(zarr_url: str, bd_channel_labels: tuple):
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)

    df = open_ome_zarr_container(zarr_url).get_feature_table(TABLE_NAME).dataframe
    assert df.index.name == "label"
    expected = {
        f"{prop}-{channel}"
        for prop in INTENSITY_PROPERTIES
        for channel in bd_channel_labels
    }
    assert expected <= set(df.columns)
    # No leftover intensity columns suffixed with the channel index
    leftover = {
        f"{prop}-{i}"
        for prop in INTENSITY_PROPERTIES
        for i in range(len(bd_channel_labels))
    }
    assert not leftover & set(df.columns)

    # BD-like features: one column per channel, or per channel pair
    pairs = list(itertools.combinations(bd_channel_labels, 2))
    fluo_pairs = list(itertools.combinations(bd_channel_labels[3:], 2))
    bd_columns = {
        f"{prop}-{channel}"
        for prop in ("intensity_total", "centroid_weighted_delta", "radial_moment")
        for channel in bd_channel_labels
    }
    bd_columns |= {f"delta_com-{a}-{b}" for a, b in pairs}
    bd_columns |= {f"correlation-{a}-{b}" for a, b in fluo_pairs}
    assert bd_columns | {"axis_ratio"} <= set(df.columns)
    assert len([col for col in df.columns if col.startswith("correlation-")]) == 3


def test_features_match_image(zarr_url: str, bd_channel_labels: tuple):
    """One row per object, with features computed on the image.

    Morphology features are in physical units, from the (y, x) pixel size.
    """
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)

    ome_zarr = open_ome_zarr_container(zarr_url)
    df = ome_zarr.get_feature_table(TABLE_NAME).dataframe
    image_ngio = ome_zarr.get_image()
    image = image_ngio.get_as_numpy(axes_order="yxc")
    pixel_size = np.array([image_ngio.pixel_size.y, image_ngio.pixel_size.x])
    label = ome_zarr.get_label(LABEL_NAME).get_as_numpy(axes_order="yx")

    label_ids = sorted(set(np.unique(label).tolist()) - {0})
    assert len(label_ids) > 0
    assert sorted(df.index.astype(int).tolist()) == label_ids

    for label_id in label_ids:
        mask = label == label_id
        row = df.loc[df.index.astype(int) == label_id].iloc[0]
        assert row["area"] == pytest.approx(mask.sum() * pixel_size.prod())
        # Major axis from the physical pixel coordinates. With anisotropic
        # pixels, this also checks the (y, x) order of the spacing
        coords = np.argwhere(mask) * pixel_size
        variance = np.linalg.eigvalsh(np.cov(coords.T, bias=True)).max()
        assert row["axis_major_length"] == pytest.approx(4 * np.sqrt(variance))
        assert row["axis_ratio"] == pytest.approx(
            row["axis_minor_length"] / row["axis_major_length"]
        )
        # Weighted centroids from skimage, in physical units
        region = measure.regionprops(
            mask.astype(int),
            intensity_image=image,
            spacing=pixel_size,
        )[0]
        centroids = np.asarray(region.centroid_weighted_local).T  # (C, yx)
        a, b = bd_channel_labels[0], bd_channel_labels[3]
        assert row[f"delta_com-{a}-{b}"] == pytest.approx(
            np.linalg.norm(centroids[0] - centroids[3]), rel=1e-4
        )
        pixels = image[mask].astype(np.float64)  # (N, C)
        for c, channel in enumerate(bd_channel_labels):
            assert row[f"intensity_mean-{channel}"] == pytest.approx(
                pixels[:, c].mean(), rel=1e-5, abs=1e-9
            )
            assert row[f"intensity_max-{channel}"] == pytest.approx(
                pixels[:, c].max(), rel=1e-5, abs=1e-9
            )
            assert row[f"intensity_min-{channel}"] == pytest.approx(
                pixels[:, c].min(), rel=1e-5, abs=1e-9
            )
            assert row[f"intensity_median-{channel}"] == pytest.approx(
                np.median(pixels[:, c]), rel=1e-5, abs=1e-9
            )
            assert row[f"intensity_total-{channel}"] == pytest.approx(
                pixels[:, c].sum(), rel=1e-5, abs=1e-9
            )
            assert row[f"centroid_weighted_delta-{channel}"] == pytest.approx(
                np.linalg.norm(centroids[c] - region.centroid_local), rel=1e-4
            )


def test_no_overwrite(zarr_url: str):
    region_props_features_task(zarr_url=zarr_url, label_image_name=LABEL_NAME)
    with pytest.raises(FileExistsError):
        region_props_features_task(
            zarr_url=zarr_url, label_image_name=LABEL_NAME, overwrite=False
        )
