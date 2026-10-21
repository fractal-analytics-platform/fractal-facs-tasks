import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tifffile
from ngio import open_ome_zarr_container, open_ome_zarr_plate

from fcf_ome_zarr.converters.fcf_s8_converter import fcf_s8_converter


def run_converter(tmp_path: Path, tiff_dir: Path, **kwargs) -> Path:
    """Run the converter without a plate and return the mosaic zarr url."""
    result = fcf_s8_converter(
        tiff_folder_path=str(tiff_dir),
        zarr_dir=str(tmp_path),
        fake_a_plate=False,
        **kwargs,
    )
    assert result is not None
    zarr_url = Path(result["image_list_updates"][0]["zarr_url"])
    assert zarr_url.is_relative_to(tmp_path)
    return zarr_url


@pytest.mark.parametrize("add_z_singleton", [True, False])
def test_converter_writes_mosaic_image(
    tmp_path: Path, example_data_dir: Path, add_z_singleton: bool
):
    zarr_url = run_converter(
        tmp_path, example_data_dir, add_z_singleton=add_z_singleton
    )
    image = open_ome_zarr_container(zarr_url).get_image()

    # 4 tiles of (73, 104) pixels in a 2 x 2 grid
    if add_z_singleton:
        assert image.axes == ("c", "z", "y", "x")
        assert image.shape == (6, 1, 146, 208)
    else:
        assert image.axes == ("c", "y", "x")
        assert image.shape == (6, 146, 208)
    assert image.dtype == "float32"
    assert image.pixel_size.y == pytest.approx(60 / 104 / 1.2)
    assert image.pixel_size.x == pytest.approx(60 / 104)


@pytest.mark.parametrize("add_z_singleton", [True, False])
def test_fov_rois_match_source_tiffs(
    tmp_path: Path, example_data_dir: Path, add_z_singleton: bool
):
    zarr_url = run_converter(
        tmp_path, example_data_dir, add_z_singleton=add_z_singleton
    )
    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    rois = ome_zarr.get_roi_table("FOV_ROI_table").rois()

    tif_paths = sorted(example_data_dir.glob("*.tif*"))
    assert [roi.name for roi in rois] == [str(i) for i in range(len(tif_paths))]

    n_nonzero = 0
    for roi, tif_path in zip(rois, tif_paths, strict=True):
        img = tifffile.imread(tif_path)
        roi_data = image.get_roi_as_numpy(roi)
        if add_z_singleton:
            roi_data = roi_data[:, 0]
        assert np.array_equal(roi_data, img)
        n_nonzero += np.count_nonzero(img)

    # Everything outside the ROIs is zero padding
    assert np.count_nonzero(image.get_as_numpy()) == n_nonzero


def test_converter_writes_image_inside_fake_plate(
    tmp_path: Path, example_data_dir: Path
):
    result = fcf_s8_converter(
        tiff_folder_path=str(example_data_dir),
        zarr_dir=str(tmp_path),
        fake_a_plate=True,
    )
    assert result is not None
    # The plate is named after the TIFF folder by default
    plate_url = f"{example_data_dir.name}.ome.zarr"
    (image_update,) = result["image_list_updates"]
    assert image_update["zarr_url"] == str(tmp_path / plate_url / "A/01/0")
    assert image_update["attributes"] == {"plate": plate_url, "well": "A01"}

    plate = open_ome_zarr_plate(tmp_path / plate_url)
    assert plate.wells_paths() == ["A/01"]
    assert plate.images_paths() == ["A/01/0"]

    image = open_ome_zarr_container(image_update["zarr_url"]).get_image()
    assert image.shape == (6, 1, 146, 208)


@pytest.mark.parametrize("zarr_name", [None, "my_mosaic"])
def test_zarr_name_sets_output_name(
    tmp_path: Path, example_data_dir: Path, zarr_name: str | None
):
    zarr_url = run_converter(tmp_path, example_data_dir, zarr_name=zarr_name)

    expected_name = zarr_name or example_data_dir.name
    assert zarr_url == tmp_path / f"{expected_name}.ome.zarr"


def test_converter_writes_roi_tables(tmp_path: Path, example_data_dir: Path):
    ome_zarr = open_ome_zarr_container(run_converter(tmp_path, example_data_dir))

    assert set(ome_zarr.list_tables()) == {"FOV_ROI_table", "image_ROI_table"}
    assert len(ome_zarr.get_roi_table("FOV_ROI_table").rois()) == 4
    assert len(ome_zarr.get_roi_table("image_ROI_table").rois()) == 1


def test_converter_writes_channel_metadata(
    tmp_path: Path, example_data_dir: Path, example_channel_labels: tuple[str, ...]
):
    image = open_ome_zarr_container(
        run_converter(tmp_path, example_data_dir)
    ).get_image()

    assert tuple(image.channel_labels) == example_channel_labels
    data = image.get_as_numpy()
    for c, channel in enumerate(image.channels_meta.channels):
        window = channel.channel_visualisation
        # Windows are computed from the data, so they lie within its value range
        assert data[c].min() <= window.start < window.end <= data[c].max()


def test_convert_first_x_tiffs_skips_remaining_files(
    tmp_path: Path, mixed_tiff_dir: Path
):
    # The mismatching TIFF sorts last, converting it would raise
    # Update this if the example dataset changes
    zarr_url = run_converter(tmp_path, mixed_tiff_dir, convert_first_x_tiffs=4)
    image = open_ome_zarr_container(zarr_url).get_image()

    # 4 tiles of (73, 104) pixels in a 2 x 2 grid
    assert image.shape == (6, 1, 146, 208)


def test_mixed_folder_raises(tmp_path: Path, mixed_tiff_dir: Path):
    with pytest.raises(ValueError, match="mismatching channels"):
        fcf_s8_converter(tiff_folder_path=str(mixed_tiff_dir), zarr_dir=str(tmp_path))


def test_converter_with_single_channel_tiffs(
    tmp_path: Path, make_tiff: Callable[..., Path]
):
    tiff_dir = tmp_path / "tiffs"
    tiff_dir.mkdir()
    for name in ("a.tif", "b.tif"):
        make_tiff(f"tiffs/{name}", shape=(1, 20, 30))

    image = open_ome_zarr_container(run_converter(tmp_path, tiff_dir)).get_image()

    # 2 tiles of (20, 30) pixels in a 1 x 2 grid
    assert image.shape == (1, 1, 20, 60)


def test_zarr_name_from_folder_with_spaces(tmp_path: Path, example_data_dir: Path):
    tiff_dir = tmp_path / "Demo recording"
    shutil.copytree(example_data_dir, tiff_dir)

    zarr_url = run_converter(tmp_path, tiff_dir)

    assert zarr_url == tmp_path / "Demo_recording.ome.zarr"
