import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

BD_S8_DIR = Path(__file__).parent / "example_data" / "BD_S8"
BD_A8_DIR = Path(__file__).parent / "example_data" / "BD_A8"

# Paths used for fixtures for the integration tests.
# Run only on S8 data for now because the OME-Zarr structure is the same.
BD_S8_TIFF_DIR = BD_S8_DIR / "Demo_data_S8_img_extractor"
BD_S8_FCS_PATH = BD_S8_DIR / "Demo recording_10000.fcs"


@dataclass(frozen=True)
class ExampleBDDataset:
    """Example recording with 4 TIFFs and the matching reduced FCS file."""

    tiff_dir: Path
    fcs_path: Path
    # (y, x) size of the 2 x 2 mosaic, set by the largest TIFF
    mosaic_size: tuple[int, int]
    # Raw values of event 1 in the full FCS export
    fcs_event_1: dict[str, float]


EXAMPLE_BD_DATASETS = {
    # TIFFs from the standalone image extractor: one 3D (c, y, x) series per file
    "BD_S8": ExampleBDDataset(
        tiff_dir=BD_S8_TIFF_DIR,
        fcs_path=BD_S8_FCS_PATH,
        mosaic_size=(146, 208),
        fcs_event_1={
            "Size (FSC)": 1179,
            "SpectralEventWidth": 4675,
            "Time": 0.41319016,
        },
    ),
    # TIFFs from the FlowJo plugin: one 2D (y, x) series per page
    "BD_A8": ExampleBDDataset(
        tiff_dir=BD_A8_DIR / "B01-Exp unstimulated fully stained_001_images_flowjo",
        fcs_path=BD_A8_DIR / "B01-Exp unstimulated fully stained_001.fcs",
        mosaic_size=(178, 208),
        fcs_event_1={
            "Size (FSC)": 1410,
            "SpectralEventWidth": 5684,
            "Time": 4.32101488,
        },
    ),
}


@pytest.fixture(params=list(EXAMPLE_BD_DATASETS))
def example_bd_dataset(request: pytest.FixtureRequest) -> ExampleBDDataset:
    """Each example dataset (BD S8 and BD A8) in turn."""
    return EXAMPLE_BD_DATASETS[request.param]


@pytest.fixture
def bd_s8_tiff_dir() -> Path:
    """S8 example recording, with TIFFs in subfolders as written by the instrument."""
    return BD_S8_TIFF_DIR


@pytest.fixture
def bd_s8_fcs_path() -> Path:
    """FCS file of the S8 example recording, reduced to the 4 imaged events."""
    return BD_S8_FCS_PATH


@pytest.fixture
def bd_channel_labels() -> tuple[str, ...]:
    """Sanitized channel labels from the PageName tags of the example TIFFs.

    The S8 and A8 example TIFFs have the same channels. The raw tags are e.g.
    "LightLoss (Imaging)" and "Imaging_BP/534/46/LP/505".
    """
    return (
        "LightLoss_Imaging",
        "FSC",
        "SSC_Imaging",
        "Imaging_BP_534_46_LP_505",
        "Imaging_BP_598_60_LP_570",
        "Imaging_BP_788_225_LP_675",
    )


@pytest.fixture
def mixed_tiff_dir(tmp_path: Path) -> Path:
    """Copy of the example data plus a single-page TIFF without metadata."""
    mixed_dir = tmp_path / "mixed_tiffs"
    shutil.copytree(BD_S8_TIFF_DIR, mixed_dir)
    tifffile.imwrite(mixed_dir / "bare.tif", np.zeros((20, 30), dtype=np.uint8))
    return mixed_dir


@pytest.fixture
def make_tiff(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing a multi-page TIFF, one page per channel.

    The pixel size is given as (y, x) in micrometers. If None, the TIFF is
    written without resolution metadata. If page_names is given, each page
    gets the matching PageName tag (one name per channel).
    """

    def _make_tiff(
        name: str,
        shape: tuple[int, int, int] = (6, 20, 30),
        pixel_size: tuple[float, float] | None = (2.0, 4.0),
        page_names: Sequence[str] | None = None,
    ) -> Path:
        tif_path = tmp_path / name
        kwargs = {}
        if pixel_size is not None:
            # Resolution is stored as (x, y) pixels per centimeter
            kwargs["resolution"] = (1e4 / pixel_size[1], 1e4 / pixel_size[0])
            kwargs["resolutionunit"] = "CENTIMETER"
        data = np.zeros(shape, dtype=np.float32)
        if page_names is None:
            tifffile.imwrite(tif_path, data, photometric="minisblack", **kwargs)
            return tif_path

        with tifffile.TiffWriter(tif_path) as tif:
            for page, page_name in zip(data, page_names, strict=True):
                tif.write(
                    page,
                    photometric="minisblack",
                    # 285 is the PageName tag
                    extratags=[(285, "s", 0, page_name, True)],
                    **kwargs,
                )
        return tif_path

    return _make_tiff
