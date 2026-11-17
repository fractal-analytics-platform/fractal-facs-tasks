"""Tests of the `.cvw` reader on local recordings.

The recordings are too large to commit, so they are read from the gitignored
`scratch/` folder. Tests are marked `local_data` and skipped when the files are
missing; CI excludes them with `-m "not local_data"`.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from fcf_ome_zarr.readers.cvw import CvwFile, reconstruct_event

pytestmark = pytest.mark.local_data

SCRATCH_DIR = Path(__file__).parents[1] / "scratch"

# Maximum allowed max|ours - tiff| / max|tiff| per event. The vendor tool rounds
# its phase calculations in float32, which we do not reproduce: the worst error
# over all exported events of the three recordings is 3e-5.
MAX_RELATIVE_ERROR = 5e-5

# Number of events compared against the TIFFs, spread over the exported range
N_COMPARED_EVENTS = 5

CHANNEL_NAMES = (
    "LightLoss (Imaging)",
    "FSC",
    "SSC (Imaging)",
    "Imaging_BP/534/46/LP/505",
    "Imaging_BP/598/60/LP/570",
    "Imaging_BP/788/225/LP/675",
)


@dataclass(frozen=True)
class CvwDataset:
    """Local recording with its `.cvw` and vendor TIFF export."""

    name: str
    tiff_dir_name: str
    n_columns: int
    has_compensation: bool

    @property
    def dir(self) -> Path:
        return SCRATCH_DIR / self.name

    @property
    def cvw_path(self) -> Path:
        return self.dir / f"{self.name}.cvw"

    @property
    def tiff_dir(self) -> Path:
        return self.dir / self.tiff_dir_name


CVW_DATASETS = [
    CvwDataset(
        name="Demo recording_10000",
        tiff_dir_name="Demo recording_10000_images_20250925_1247_47_extractor_old",
        n_columns=8,
        has_compensation=False,
    ),
    CvwDataset(
        name="PE_Smoothing_0 Pixel Threshold 1_LowThresholdSSC",
        tiff_dir_name=(
            "PE_Smoothing_0 Pixel Threshold 1_LowThresholdSSC"
            "_images_20261116_1151_19_standalone"
        ),
        n_columns=8,
        has_compensation=False,
    ),
    CvwDataset(
        name="B01-Exp unstimulated fully stained_001",
        tiff_dir_name=(
            "B01-Exp unstimulated fully stained_001_images_20261111_0928_27_extractor"
        ),
        n_columns=6,
        has_compensation=True,
    ),
]


@pytest.fixture(params=CVW_DATASETS, ids=lambda d: d.name)
def dataset(request: pytest.FixtureRequest) -> CvwDataset:
    dataset: CvwDataset = request.param
    for path in (dataset.cvw_path, dataset.tiff_dir):
        if not path.exists():
            pytest.skip(f"Local data not found: {path}")
    return dataset


@pytest.fixture
def cvw(dataset: CvwDataset) -> CvwFile:
    return CvwFile(dataset.cvw_path)


def tiff_paths_by_event(tiff_dir: Path) -> dict[int, Path]:
    """Map event index to TIFF path, from the `_<NNNNNNNN>.tiff` suffix."""
    paths = {}
    for path in tiff_dir.rglob("*.tif*"):
        match = re.search(r"_(\d{8})\.tiff?$", path.name)
        if match:
            paths[int(match.group(1))] = path
    return paths


def compared_events(tiff_dir: Path) -> list[tuple[int, Path]]:
    """Events spread evenly over the exported range, with their TIFF paths."""
    paths = tiff_paths_by_event(tiff_dir)
    events = sorted(paths)
    picks = np.linspace(0, len(events) - 1, N_COMPARED_EVENTS).round().astype(int)
    return [(events[p], paths[events[p]]) for p in picks]


def read_tiff(path: Path) -> np.ndarray:
    """Read a vendor TIFF as (c, y, x), one page per channel."""
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


def test_parses_header_and_metadata(dataset: CvwDataset, cvw: CvwFile):
    assert cvw.n_columns == dataset.n_columns
    assert len(cvw.event_numbers) == cvw.n_events
    assert cvw.channel_names == CHANNEL_NAMES
    assert (cvw.compensation_matrix is not None) == dataset.has_compensation


def test_reconstruction_matches_tiff(dataset: CvwDataset, cvw: CvwFile):
    for event, tiff_path in compared_events(dataset.tiff_dir):
        expected = read_tiff(tiff_path)
        image = reconstruct_event(cvw, event)

        assert cvw.image_shape(event) == expected.shape[1:]
        assert image.shape == expected.shape
        assert image.dtype == np.float32
        error = np.abs(image - expected).max() / np.abs(expected).max()
        assert error <= MAX_RELATIVE_ERROR, f"event {event}: relative error {error}"
