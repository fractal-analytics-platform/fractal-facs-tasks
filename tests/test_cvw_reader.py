"""Tests of the `.cvw` reader on local recordings, see `tests/local_data.py`."""

import numpy as np
import pytest

from fcf_ome_zarr.readers.cvw import CvwFile
from tests.local_data import (
    CVW_DATASETS,
    MAX_RELATIVE_ERROR,
    CvwDataset,
    compared_events,
    read_tiff,
    relative_error,
)

pytestmark = pytest.mark.local_data

CHANNEL_NAMES = (
    "LightLoss (Imaging)",
    "FSC",
    "SSC (Imaging)",
    "Imaging_BP/534/46/LP/505",
    "Imaging_BP/598/60/LP/570",
    "Imaging_BP/788/225/LP/675",
)


@pytest.fixture(params=CVW_DATASETS, ids=lambda d: d.name)
def dataset(request: pytest.FixtureRequest) -> CvwDataset:
    dataset: CvwDataset = request.param
    dataset.skip_if_missing()
    return dataset


@pytest.fixture
def cvw(dataset: CvwDataset) -> CvwFile:
    return CvwFile(dataset.cvw_path)


def test_parses_header_and_metadata(dataset: CvwDataset, cvw: CvwFile):
    assert cvw.n_columns == dataset.n_columns
    assert len(cvw.event_numbers) == cvw.n_events
    assert cvw.channel_names == CHANNEL_NAMES
    assert (cvw.compensation_matrix is not None) == dataset.has_compensation


def test_reconstruction_matches_tiff(dataset: CvwDataset, cvw: CvwFile):
    for event, tiff_path in compared_events(dataset.tiff_dir):
        expected = read_tiff(tiff_path)
        image = cvw.read_image(event)

        assert cvw.image_shape(event) == expected.shape[1:]
        assert image.shape == expected.shape
        assert image.dtype == np.float32
        error = relative_error(image, expected)
        assert error <= MAX_RELATIVE_ERROR, f"event {event}: relative error {error}"
