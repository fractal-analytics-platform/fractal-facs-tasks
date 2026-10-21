"""Mosaic layout shared by the converters."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import tifffile
from ngio import RoiSlice
from numpy.typing import DTypeLike

from fcf_ome_zarr.utils import sanitize_string


# Might shadow PixelSize from ngio if not careful, keep for now
class PixelSize(NamedTuple):
    """Pixel size of the image in micrometers"""

    y: float
    x: float


class TileSize(NamedTuple):
    """Tile size of the mosaic image, equal to max (y, x) size across all TIFF files"""

    y: int
    x: int


class MosaicSize(NamedTuple):
    """Size of the mosaic OME-Zarr in pixels"""

    y: int
    x: int


@dataclass(frozen=True)
class MosaicLayout:
    """Class storing the mosaic layout information"""

    tile: TileSize
    pixel_size: PixelSize
    n_tiles: int
    channel_labels: tuple[str, ...]

    @cached_property
    def n_channels(self) -> int:
        """Number of channels of the mosaic"""
        return len(self.channel_labels)

    @cached_property
    def n_cols(self) -> int:
        """Number of columns, chosen to make the mosaic as square as possible"""
        return math.ceil(math.sqrt(self.n_tiles * self.tile.y / self.tile.x))

    @cached_property
    def n_rows(self) -> int:
        """Number of rows needed to fit all tiles"""
        return math.ceil(self.n_tiles / self.n_cols)

    @cached_property
    def size(self) -> MosaicSize:
        """Total (y, x) size of the mosaic OME-Zarr, in pixels"""
        return MosaicSize(y=self.n_rows * self.tile.y, x=self.n_cols * self.tile.x)

    @classmethod
    def from_paths(cls, tif_paths: Sequence[Path]) -> Self:
        """Initialize MosaicLayout from a sequence of TIFF file paths"""
        if not tif_paths:
            raise ValueError("No TIFF files provided to build the mosaic layout")

        # Tile size is the maximum (y, x) size across all TIFF files
        max_y = 0
        max_x = 0
        channel_labels: set[tuple[str, ...]] = set()
        pixel_size: set[PixelSize] = set()
        for tif_path in tif_paths:
            with tifffile.TiffFile(tif_path) as tif:
                page = tif.pages.first
                max_y = max(max_y, page.imagelength)
                max_x = max(max_x, page.imagewidth)
                channel_labels.add(_get_channel_labels(tif=tif))
                pixel_size.add(_get_pixel_size(tif=tif))

        # Check that all images have the same number of channels and pixel size
        if len(channel_labels) != 1:
            raise ValueError("The provided TIFF images have mismatching channels")
        if len(pixel_size) != 1:
            raise ValueError("The provided TIFF images have mismatching pixel size")

        return cls(
            tile=TileSize(y=max_y, x=max_x),
            n_tiles=len(tif_paths),
            channel_labels=channel_labels.pop(),
            pixel_size=pixel_size.pop(),
        )

    def _get_tile_origin(self, row: int, col: int) -> tuple[int, int]:
        """Get (y, x) origin of a tile at given row and col indexer"""
        return row * self.tile.y, col * self.tile.x

    def get_tile_slice(
        self, row: int, col: int, add_z_singleton: bool = False
    ) -> tuple[slice | int, ...]:
        """Get slice for indexing tile at given row and col indexer"""
        y0, x0 = self._get_tile_origin(row=row, col=col)
        if add_z_singleton:
            indexer = np.s_[:, 0, y0 : y0 + self.tile.y, x0 : x0 + self.tile.x]
        else:
            indexer = np.s_[:, y0 : y0 + self.tile.y, x0 : x0 + self.tile.x]
        return indexer

    def get_roi_slices(
        self, row: int, col: int, img_shape: tuple[int, int]
    ) -> tuple[RoiSlice, ...]:
        """Get ROI Slice for an image of shape img_shape at given row and col indexer"""
        y0, x0 = self._get_tile_origin(row=row, col=col)
        roi_slices = (
            RoiSlice(
                axis_name="x",
                start=x0 * self.pixel_size.x,
                length=img_shape[1] * self.pixel_size.x,
            ),
            RoiSlice(
                axis_name="y",
                start=y0 * self.pixel_size.y,
                length=img_shape[0] * self.pixel_size.y,
            ),
        )
        return roi_slices


def _get_pixel_size(tif: tifffile.TiffFile) -> PixelSize:
    """Return the (y, x) pixel size in micrometers of a BD S8 TIFF.

    The absolute values of the XResolution/YResolution tags written by the S8 are
    off by orders of magnitude, so they are not used directly. Instead, the image
    width is assumed to span the fixed 60 µm beam path of the instrument (104 spots
    across 60 µm, see 10.1126/science.abj3013), and only the ratio of the
    resolution tags is used to derive the y pixel size from the x pixel size.

    NOTE: this is specific to the BD S8 instrument. It is a candidate for a
    per-instrument implementation if more instruments are supported.
    """
    # Width of the beam path imaged across x, in micrometers
    image_width_um = 60.0

    page = tif.pages.first
    # Only the y/x anisotropy of the resolution tags is reliable
    x_res, y_res = page.get_resolution()
    pixel_size_x = image_width_um / page.imagewidth
    pixel_size = PixelSize(y=pixel_size_x * x_res / y_res, x=pixel_size_x)
    return pixel_size


def _get_channel_labels(tif: tifffile.TiffFile) -> tuple[str, ...]:
    """Return the sanitized channel labels from the PageName tag of each tif page.

    Labels are sanitized with `sanitize_string`. Pages without a PageName tag,
    or whose label is empty after sanitizing, fall back to "channel_{i}".
    Raises a ValueError if two labels are identical after sanitizing.
    
    NOTE: this is specific to the BD S8 instrument. It is a candidate for a
    per-instrument implementation if more instruments are supported.
    """
    labels: list[str] = []
    raw_labels: dict[str, str] = {}
    for i, page in enumerate(tif.pages):
        # aspage() narrows TiffPage | TiffFrame to TiffPage, which holds the tags
        page_name = page.aspage().tags.get("PageName")
        raw_label = str(page_name.value) if page_name is not None else ""
        label = sanitize_string(raw_label) or f"channel_{i}"
        if label in raw_labels:
            raise ValueError(
                f"Channel labels {raw_labels[label]!r} and {raw_label!r} in "
                f"{tif.filename} are both sanitized to {label!r}."
            )
        raw_labels[label] = raw_label
        labels.append(label)
    return tuple(labels)


def pad_to_shape(
    img: np.ndarray, target_shape: tuple[int, int], dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Pad a 3D image (c, y, x) to the target shape (y, x) with zeros."""
    c, y, x = img.shape
    padded = np.zeros((c, *target_shape), dtype=dtype)
    padded[:, :y, :x] = img
    return padded
