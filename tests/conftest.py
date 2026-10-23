import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
import tifffile

EXAMPLE_DATA_DIR = Path(__file__).parent / "example_data" / "Demo_data_S8"
# First 4 events (matching the TIFFs) and 8 parameters of the full FCS export,
# written by example_data/make_test_fcs.py
EXAMPLE_FCS_PATH = Path(__file__).parent / "example_data" / "Demo recording_10000.fcs"


@pytest.fixture
def example_data_dir() -> Path:
    """S8 example recording, with TIFFs in subfolders as written by the instrument."""
    return EXAMPLE_DATA_DIR


@pytest.fixture
def example_fcs_path() -> Path:
    """FCS file of the S8 example recording, reduced to the 4 imaged events."""
    return EXAMPLE_FCS_PATH


@pytest.fixture
def example_channel_labels() -> tuple[str, ...]:
    """Sanitized channel labels from the PageName tags of the S8 example TIFFs.

    The raw tags are e.g. "LightLoss (Imaging)" and "Imaging_BP/534/46/LP/505".
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
    shutil.copytree(EXAMPLE_DATA_DIR, mixed_dir)
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
