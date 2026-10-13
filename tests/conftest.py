import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tifffile

EXAMPLE_DATA_DIR = Path(__file__).parent / "example_data"


@pytest.fixture
def example_data_dir() -> Path:
    """Folder with the S8 example TIFFs."""
    return EXAMPLE_DATA_DIR


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
    written without resolution metadata.
    """

    def _make_tiff(
        name: str,
        shape: tuple[int, int, int] = (6, 20, 30),
        pixel_size: tuple[float, float] | None = (2.0, 4.0),
    ) -> Path:
        tif_path = tmp_path / name
        kwargs = {}
        if pixel_size is not None:
            # Resolution is stored as (x, y) pixels per centimeter
            kwargs["resolution"] = (1e4 / pixel_size[1], 1e4 / pixel_size[0])
            kwargs["resolutionunit"] = "CENTIMETER"
        tifffile.imwrite(
            tif_path,
            np.zeros(shape, dtype=np.float32),
            photometric="minisblack",
            **kwargs,
        )
        return tif_path

    return _make_tiff
