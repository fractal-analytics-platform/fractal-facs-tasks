"""Smoke tests of the converter with `.cvw` input, see `tests/local_data.py`.

The main test converts a whole recording (~10000 events) and spot-checks a few
ROIs against the vendor TIFFs. The mosaic machinery is shared with the TIFF input
and tested in `test_fcf_bd_converter.py`.
"""

from pathlib import Path

import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.converters.fcf_bd_converter import fcf_bd_converter
from fcf_ome_zarr.converters.utils import AdvancedBDConverterParameters
from fcf_ome_zarr.readers.cvw import CvwFile
from fcf_ome_zarr.utils import sanitize_string
from tests.local_data import (
    CVW_DATASETS,
    MAX_RELATIVE_ERROR,
    CvwDataset,
    compared_events,
    read_tiff,
    relative_error,
)

pytestmark = pytest.mark.local_data


@pytest.mark.parametrize("dataset", CVW_DATASETS, ids=lambda d: d.name)
def test_cvw_input_matches_tiffs(tmp_path: Path, dataset: CvwDataset):
    dataset.skip_if_missing()
    result = fcf_bd_converter(
        input_path=str(dataset.cvw_path),
        zarr_dir=str(tmp_path),
        advanced_parameters=AdvancedBDConverterParameters(convert_to_uint16=False),
    )
    assert result is not None
    zarr_url = result["image_list_updates"][0]["zarr_url"]
    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    rois = ome_zarr.get_roi_table("FOV_ROI_table").rois()

    # Named after the .cvw file, one ROI per event, pixel size of the TIFF exports
    assert Path(zarr_url).name == f"{sanitize_string(dataset.name)}.ome.zarr"
    n_events = CvwFile(dataset.cvw_path).n_events
    assert [roi.name for roi in rois] == [str(i) for i in range(n_events)]
    assert image.pixel_size.y == pytest.approx(60 / 104 / 1.2)
    assert image.pixel_size.x == pytest.approx(60 / 104)

    for event, tiff_path in compared_events(dataset.tiff_dir):
        expected = read_tiff(tiff_path)
        roi_data = image.get_roi_as_numpy(rois[event])

        assert roi_data.shape == expected.shape
        error = relative_error(roi_data, expected)
        assert error <= MAX_RELATIVE_ERROR, f"event {event}: relative error {error}"


@pytest.mark.parametrize("dataset", CVW_DATASETS, ids=lambda d: d.name)
def test_convert_first_n_events_skips_remaining_events(
    tmp_path: Path, dataset: CvwDataset
):
    dataset.skip_if_missing()
    result = fcf_bd_converter(
        input_path=str(dataset.cvw_path),
        zarr_dir=str(tmp_path),
        advanced_parameters=AdvancedBDConverterParameters(
            fake_a_plate=False, convert_first_n_events=3
        ),
    )
    assert result is not None
    zarr_url = result["image_list_updates"][0]["zarr_url"]
    rois = open_ome_zarr_container(zarr_url).get_roi_table("FOV_ROI_table").rois()

    assert [roi.name for roi in rois] == ["0", "1", "2"]
