"""This is the Python module for my_task."""

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Optional, Self

import ngio
import numpy as np
import tifffile
import zarr
from dask.array.core import Array as DaskArray
from ngio import NgioSupportedStore, Roi, RoiSlice
from ngio.tables import RoiTable
from numpy.typing import DTypeLike
from pydantic import validate_call
from zarr.storage import LocalStore


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
    n_channels: int

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
        n_channels: set[int] = set()
        pixel_size: set[PixelSize] = set()
        for tif_path in tif_paths:
            with tifffile.TiffFile(tif_path) as tif:
                page = tif.pages.first
                max_y = max(max_y, page.imagelength)
                max_x = max(max_x, page.imagewidth)
                n_channels.add(len(tif.pages))
                pixel_size.add(_get_pixel_size(tif=tif))

        # Check that all images have the same number of channels and pixel size
        if len(n_channels) != 1:
            raise ValueError("The provided TIFF images have mismatching channels")
        if len(pixel_size) != 1:
            raise ValueError("The provided TIFF images have mismatching pixel size")

        return cls(
            tile=TileSize(y=max_y, x=max_x),
            n_tiles=len(tif_paths),
            n_channels=n_channels.pop(),
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
    ) -> list[RoiSlice]:
        """Get ROI Slice for an image of shape img_shape at given row and col indexer"""
        y0, x0 = self._get_tile_origin(row=row, col=col)
        roi_slices = [
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
        ]
        return roi_slices


def _get_pixel_size(tif: tifffile.TiffFile) -> PixelSize:
    """Return the (y, x) pixel size in micrometers from the tif resolution tags."""
    page = tif.pages.first
    # Assume for now that all tifs are well-behaved
    # Get resolution in pixels per micrometer
    x_res, y_res = page.get_resolution(unit=tifffile.RESUNIT.MICROMETER)
    # Pixel size in micrometer
    pixel_size = PixelSize(y=1 / y_res, x=1 / x_res)
    return pixel_size


def _pad_to_shape(
    img: np.ndarray, target_shape: tuple[int, int], dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Pad a 3D image (c, y, x) to the target shape (y, x) with zeros."""
    c, y, x = img.shape
    padded = np.zeros((c, *target_shape), dtype=dtype)
    padded[:, :y, :x] = img
    return padded


def sampled_percentiles(
    ac: DaskArray, low: float = 0.5, high: float = 99.5, n_chunks: int = 200
) -> np.ndarray:
    """Estimate percentiles from a sample of random chunks."""
    # pick random chunks
    all_chunks = list(np.ndindex(*ac.numblocks))
    chosen = random.sample(all_chunks, min(len(all_chunks), n_chunks))
    samples = []
    for idx in chosen:
        block = ac.blocks[idx].compute()
        samples.append(block.ravel())
    samples = np.concatenate(samples)
    return np.percentile(samples, [low, high])


def reset_omero_channels(store: NgioSupportedStore, level: str = "0") -> None:
    """Custom OMERO channel metadata reset for S8 converter"""
    ome_zarr_container = ngio.open_ome_zarr_container(store)
    img = ome_zarr_container.get_image(path=level)
    arr_dask = img.get_as_dask()  # (C, Y, X) float32 in your case

    # Compute robust ranges; adjust as you like
    p_lo, p_hi = 0.05, 99.995
    starts, ends, mins, maxs = [], [], [], []
    for c in range(arr_dask.shape[0]):
        ac = arr_dask[c]
        lo, hi = sampled_percentiles(ac, p_lo, p_hi)
        mn, mx = ac.min().compute(), ac.max().compute()
        starts.append(float(lo))
        ends.append(float(hi))
        mins.append(float(mn))
        maxs.append(float(mx))

    # Optional: supply labels/colors/active; otherwise leave as None
    labels = ["Lightloss", "SSC", "FSC", "FL1", "FL2", "FL3"]
    colors = ["FFFFFF", "00FF00", "0000FF", "FF0000", "FF00FF", "00FFFF"]
    active = [True, True, True, True, False, False]

    # Hard-code lightloss rescaling
    starts[0] = 0.0
    ends[0] = 0.2

    # Hard-code all starts to 0
    for i in range(len(starts)):
        starts[i] = 0.0

    # Hard-code all max to 1 (to avoid validation errors)
    for i in range(len(maxs)):
        maxs[i] = 1.0

    # Write to the Zarr root attrs
    root = zarr.open_group(store, mode="r+")
    root.attrs["omero"] = {
        "channels": [
            {
                "label": labels[c],
                "color": (colors[c] if colors else None),
                "active": (active[c] if active else None),
                "window": {
                    "min": mins[c],
                    "max": maxs[c],
                    "start": starts[c],
                    "end": ends[c] + 0.1,
                },
            }
            for c in range(len(mins))
        ]
    }


@validate_call
def fcf_s8_converter(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Input parameters
    tiff_folder_path: str,
    fake_a_plate: bool = True,
    add_z_singleton: bool = True,
    convert_first_x_tiffs: Optional[int] = None,
    overwrite: bool = True,
) -> dict | None:
    """Apply a Gaussian blur to the input image and save the result as a OME-Zarr image.

    Args:
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        tiff_folder_path: Path to the folder containing the TIFF files of the
            S8 FACS to be converted.
        fake_a_plate: Whether to create a fake plate structure with a single
            well "A1" containing the mosaic image. This is useful to use some
            tools that need plates (e.g. Fractal feature explorer).
            Defaults to True.
        add_z_singleton: Whether to add a singleton Z dimension to the OME-Zarr.
            This is useful to use some of the tools (like the Fractal feature
            explorer) that require Z singleton dimensions.
            Defaults to True.
        convert_first_x_tiffs: If provided, only the first X TIFF files in the
            folder are converted. This is useful for testing with a smaller
            number of files. If None, all TIFF files in the folder are converted.
        overwrite (bool): Whether to overwrite an existing OME-Zarr image.
            Defaults to True.
    """
    logging.info(f"Processing {tiff_folder_path=}")
    tiff_folder = Path(tiff_folder_path)

    plate_url = "plate.ome.zarr"
    # Probably update name not to be hardcoded?
    image_name = "s8_cells_mosaic.ome.zarr"
    if fake_a_plate:
        zarr_url = Path(f"{zarr_dir}/{plate_url}/{image_name}")

        plate = ngio.create_empty_plate(
            f"{zarr_dir}/{plate_url}", name=plate_url, overwrite=True
        )
        well = plate.add_well(row="A", column=1)
        well.add_image(image_name)

    else:
        zarr_url = Path(f"{zarr_dir}/{image_name}")

    # Convert the TIFFs to a mosaic OME-Zarr
    # Calcualte the mosaic layout parameters and extract image metadata
    if convert_first_x_tiffs is not None:
        tif_paths = sorted(tiff_folder.glob("*.tif*"))[:convert_first_x_tiffs]
    else:
        tif_paths = sorted(tiff_folder.glob("*.tif*"))

    layout = MosaicLayout.from_paths(tif_paths)

    logging.info(
        f"Grid layout: {layout.n_rows} rows x {layout.n_cols} cols = "
        f"{layout.n_rows * layout.n_cols} slots"
    )
    logging.info(f"Total image size: {layout.size.y} y x {layout.size.x} x")

    # Create OME-Zarr
    # TODO: consider more robust alternative to local store
    store = LocalStore(zarr_url)
    ome_zarr_container = ngio.create_empty_ome_zarr(
        store,
        shape=(layout.n_channels, 1, *layout.size)
        if add_z_singleton
        else (layout.n_channels, *layout.size),
        axes_names=["c", "z", "y", "x"] if add_z_singleton else ["c", "y", "x"],
        pixelsize=layout.pixel_size,
        dtype="float32",
        chunks=(layout.n_channels, 1, 512, 512)
        if add_z_singleton
        else (layout.n_channels, 512, 512),  # adjust as needed
        overwrite=overwrite,
        ngff_version="0.5",
    )
    ngio_image = ome_zarr_container.get_image()
    zarr_array = ngio_image.zarr_array

    rois = []

    # Loop over TIFFs & place them inside the mosaic
    for i, tif_path in enumerate(tif_paths):
        row, col = divmod(i, layout.n_cols)
        tile_slice = layout.get_tile_slice(
            row=row, col=col, add_z_singleton=add_z_singleton
        )

        with tifffile.TiffFile(tif_path) as tif:
            img = tif.asarray().astype(np.float32)
            if img.ndim == 2:
                img = img[None, :, :]  # add channel if missing

        # Place padded image into the mosaic tile
        padded = _pad_to_shape(img, target_shape=layout.tile)
        zarr_array[tile_slice] = padded

        # Calculate roi slices
        roi_slices = layout.get_roi_slices(row=row, col=col, img_shape=img.shape[-2:])
        rois.append(
            Roi(
                name=str(i),  # TODO: find out how to match this with the spectra
                slices=roi_slices,
                space="world",
            )
        )

    # --- Create ROI table ---
    roi_table = RoiTable(rois=rois)
    ome_zarr_container.add_table("FOV_ROI_table", roi_table, overwrite=overwrite)

    # Genereate a ROI table that covers the full image
    image_roi_table = ngio_image.build_image_roi_table()
    ome_zarr_container.add_table(
        "image_ROI_table", image_roi_table, overwrite=overwrite
    )

    ngio_image.consolidate()
    reset_omero_channels(store, level="3")

    if fake_a_plate:
        image_list_update_dict = {
            "image_list_updates": [
                {
                    "zarr_url": str(zarr_url),
                    "attributes": {"plate": "plate.ome.zarr", "well": "A1"},
                }
            ]
        }
    else:
        image_list_update_dict = {"image_list_updates": [{"zarr_url": str(zarr_url)}]}
    return image_list_update_dict


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=fcf_s8_converter)
