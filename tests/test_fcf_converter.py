from pathlib import Path

# import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.fcf_s8_converter import (
    fcf_s8_converter,
)


def test_fcf_s8_converter(tmp_path: Path):
    """Base test for the FCF S8 converter task."""

    # Path to this test file
    TEST_DIR = Path(__file__).parent

    # Path to example_data folder
    EXAMPLE_DATA_DIR = TEST_DIR / "example_data"

    result = fcf_s8_converter(
        tiff_folder_path=str(EXAMPLE_DATA_DIR),
        zarr_dir=str(tmp_path),
        xy_pixelsize=1.0,
        fake_a_plate=True,
        add_z_singleton=True,
        overwrite=True,
    )
    assert result is not None
    output_image_url = result["image_list_updates"][0]["zarr_url"]
    assert Path(output_image_url).exists()
    ome_zarr = open_ome_zarr_container(output_image_url)
    image = ome_zarr.get_image()
    assert image is not None
