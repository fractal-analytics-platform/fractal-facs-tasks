"""Converter for the BD S8 and A8 cell sorter imaging data at UZH FCF"""

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import flowio
import ngio
import numpy as np
import polars as pl
import tifffile
from ngio import Roi
from ngio.tables import FeatureTable, RoiTable
from ngio.tables.v1._feature_table import FeatureTableMeta
from pydantic import PositiveInt, validate_call
from zarr import Array
from zarr.storage import LocalStore

from fcf_ome_zarr.constants import (
    EVENT_INDEX_KEY,
    FCS_EVENTS_TABLE,
    FCS_METADATA_FILE,
    FOV_ROI_TABLE,
    IMAGE_ROI_TABLE,
    NGFF_VERSION,
    TABLE_BACKEND,
    UINT16_MAX_INTENSITY,
)
from fcf_ome_zarr.converters.mosaic import MosaicLayout, PixelSize, TileSize
from fcf_ome_zarr.readers.cvw import IMAGE_WIDTH, CvwFile
from fcf_ome_zarr.utils import sanitize_string


@validate_call
def fcf_bd_converter(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Input parameters
    input_path: str,
    fcs_file_path: Optional[str] = None,
    zarr_name: Optional[str] = None,
    fake_a_plate: bool = True,
    add_z_singleton: bool = False,
    use_sharding: bool = False,
    convert_to_uint16: bool = False,
    target_chunk_size: int = 512,
    target_shard_size: int = 2048,
    convert_first_n_events: Optional[PositiveInt] = None,
    overwrite: bool = True,
) -> dict | None:
    """Convert BD S8 or A8 images into a single mosaic OME-Zarr image.

    The images are read either from a folder of TIFF files, extracted with the
    standalone BD image extractor or with the FlowJo plugin, or reconstructed
    directly from the raw waveforms of a `.cvw` file.

    Each image is placed in its own tile of the mosaic and recorded as a ROI in
    the "FOV_ROI_table". Channel labels are read from the TIFF PageName tags or
    from the `.cvw` metadata, with special characters (e.g. whitespace, "/",
    parentheses) replaced by underscores. If an FCS file is given, its raw
    event data is saved as the "fcs_events" feature table.

    Args:
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        input_path: Path to the BD S8 or A8 recording to convert. Either a
            folder containing the exported TIFF files, or a `.cvw` waveform
            file. With a `.cvw` file, the pixel size is computed from its
            metadata (see `CvwFile.pixel_size`).
        fcs_file_path: Path to the FCS file of the same recording. If provided,
            all its events are saved as the "fcs_events" feature table, with
            the raw values of all parameters and an "event_index" column
            matching the names of the ROIs in "FOV_ROI_table". Values are
            stored as float32, so integer parameters show up as meaningless
            floats until reinterpreted (e.g. with `.view(np.uint32)`).
            All its TEXT keywords are saved under "fcs_text" in a
            "fcs_metadata.json" file next to the image "zarr.json". Keywords are
            lowercased and without the leading "$" (as read by flowio);
            values are the unchanged strings.
        zarr_name: Name of the output OME-Zarr, without the ".ome.zarr"
            extension. Used for the plate if `fake_a_plate` is True, otherwise
            for the image. If None, the name of the TIFF folder or of the
            `.cvw` file (without extension) is used, with special characters
            (e.g. whitespace, "#", parentheses) replaced by underscores.
        fake_a_plate: Whether to create a fake plate structure with a single
            well "A01" containing the mosaic image. This is useful to use some
            tools that need plates (e.g. Fractal feature explorer).
            Defaults to True.
        add_z_singleton: Whether to add a singleton Z dimension to the OME-Zarr.
            This is useful to use some of the tools (like the Fractal feature
            explorer) that require Z singleton dimensions.
            Defaults to False.
        use_sharding: Whether to store the image in shards. If True, each chunk
            holds a single mosaic tile and shards are about
            `target_shard_size` pixels per side. If False, there is no sharding
            and chunks are about `target_chunk_size` pixels per side. Chunks
            and shards always hold all channels and a whole number of tiles.
            Defaults to False.
        convert_to_uint16: Whether to store the image as uint16 instead of
            float32. Values are clipped to [0, 8] and scaled linearly to the
            full uint16 range [0, 65535]. Defaults to False.
        target_chunk_size: Target size in pixels of the chunks along y and x,
            used only without sharding. Rounded to the closest whole number of
            tiles (at least one). This parameter is ignored if `use_sharding` is True.
            Defaults to 512.
        target_shard_size: Target size in pixels of the shards along y and x,
            used only with sharding. Rounded to the closest whole number of
            tiles (at least one). This parameter is ignored if `use_sharding` is False.
            Defaults to 2048.
        convert_first_n_events: If provided, only the first N events are
            converted: the first N TIFF files of the folder, or the first N
            events of the `.cvw` file. This is useful for testing with fewer
            events. If None, all events are converted.
        overwrite (bool): Whether to overwrite an existing OME-Zarr image (and
            fake plate, if `fake_a_plate` is True). Defaults to True.
    """
    logging.info(f"Processing {input_path=}")
    source = _open_source(Path(input_path), convert_first_n_events)
    layout = source.layout

    if zarr_name is None:
        zarr_name = sanitize_string(source.name)
    # Either the fake plate or the image, depending on fake_a_plate
    mosaic_url = f"{zarr_name}.ome.zarr"
    # Image list attributes, only set for the fake plate
    attributes: dict[str, str] = {}
    if fake_a_plate:
        plate = ngio.create_empty_plate(
            f"{zarr_dir}/{mosaic_url}",
            name=zarr_name,
            ngff_version=NGFF_VERSION,
            overwrite=overwrite,
        )
        image_path = plate.add_image(row="A", column=1, image_path="0")
        zarr_url = Path(f"{zarr_dir}/{mosaic_url}/{image_path}")
        attributes = {"plate": mosaic_url, "well": "".join(image_path.split("/")[:2])}

    else:
        zarr_url = Path(f"{zarr_dir}/{mosaic_url}")

    logging.info(
        f"Grid layout: {layout.n_rows} rows x {layout.n_cols} cols = "
        f"{layout.n_rows * layout.n_cols} slots"
    )
    logging.info(f"Total image size: {layout.size.y} y x {layout.size.x} x")

    # Chunks and shards hold all channels and a whole number of tiles
    if use_sharding:
        chunks_yx: tuple[int, int] = layout.tile
        shards_yx = layout.closest_tile_multiple(target_shard_size)
    else:
        chunks_yx = layout.closest_tile_multiple(target_chunk_size)
        shards_yx = None
    z_dims = (1,) if add_z_singleton else ()
    chunks = (layout.n_channels, *z_dims, *chunks_yx)
    shards = (layout.n_channels, *z_dims, *shards_yx) if shards_yx else None
    logging.info(f"Chunks: {chunks}, shards: {shards}")

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
        channels_meta=list(layout.channel_labels),
        dtype="uint16" if convert_to_uint16 else "float32",
        chunks=chunks,
        shards=shards,
        overwrite=overwrite,
        ngff_version=NGFF_VERSION,
    )
    ngio_image = ome_zarr_container.get_image()

    # Write whole chunks (or shards) at once, so each is written only once
    rois = _write_mosaic(
        zarr_array=ngio_image.zarr_array,
        source=source,
        block_size=shards_yx or chunks_yx,
        add_z_singleton=add_z_singleton,
        convert_to_uint16=convert_to_uint16,
    )

    # --- Create ROI table ---
    roi_table = RoiTable(rois=rois)
    ome_zarr_container.add_table(
        FOV_ROI_TABLE, roi_table, overwrite=overwrite, backend=TABLE_BACKEND
    )

    # Genereate a ROI table that covers the full image
    image_roi_table = ngio_image.build_image_roi_table()
    ome_zarr_container.add_table(
        IMAGE_ROI_TABLE, image_roi_table, overwrite=overwrite, backend=TABLE_BACKEND
    )

    if fcs_file_path is not None:
        ome_zarr_container.add_table(
            FCS_EVENTS_TABLE,
            read_fcs_events(Path(fcs_file_path)),
            overwrite=overwrite,
            backend=TABLE_BACKEND,
        )
        write_fcs_metadata(Path(fcs_file_path), zarr_url / FCS_METADATA_FILE)

    ngio_image.consolidate(mode="auto")
    # Set contrast limits for viewers
    ome_zarr_container.set_channel_windows_with_percentiles(percentiles=(0.05, 99.995))

    return {
        "image_list_updates": [{"zarr_url": str(zarr_url), "attributes": attributes}]
    }


def float_to_uint16(img: np.ndarray) -> np.ndarray:
    """Clip to [0, UINT16_MAX_INTENSITY] and scale linearly to [0, 65535]."""
    max_uint16 = np.iinfo(np.uint16).max
    scaled = np.clip(img, 0, UINT16_MAX_INTENSITY) * (max_uint16 / UINT16_MAX_INTENSITY)
    return np.round(scaled).astype(np.uint16)


@dataclass(frozen=True)
class MosaicSource:
    """Images to place in the mosaic, one per tile.

    Attributes:
        name: Default name of the output OME-Zarr, before sanitizing.
        layout: Mosaic layout, with one tile per image.
        read_image: Returns the (c, y, x) float32 image of tile `i`.
    """

    name: str
    layout: MosaicLayout
    read_image: Callable[[int], np.ndarray]


def _open_source(
    input_path: Path, convert_first_n_events: Optional[int]
) -> MosaicSource:
    """Pick the TIFF folder or the `.cvw` route, depending on the input path.

    If `convert_first_n_events` is given, only the first N events are used.
    """
    if input_path.is_dir():
        return _open_tiff_folder(input_path, convert_first_n_events)
    if input_path.suffix == ".cvw" and input_path.is_file():
        return _open_cvw_file(input_path, convert_first_n_events)
    raise ValueError(
        f"Input path {input_path} is neither a folder of TIFF files nor a .cvw file"
    )


def _open_tiff_folder(
    tiff_folder: Path, convert_first_n_events: Optional[int]
) -> MosaicSource:
    """Images from a folder of TIFF files, one tile per file in sorted order.

    TIFF images are in subdirs containing up to 10000 images each, but their ID is
    unique across the whole folder.
    """
    tif_paths = sorted(tiff_folder.rglob("*.tif*"))[:convert_first_n_events]

    def read_image(i: int) -> np.ndarray:
        with tifffile.TiffFile(tif_paths[i]) as tif:
            # Load all pages explicitly
            img = tif.asarray(key=slice(None)).astype(np.float32)
        if img.ndim == 2:
            raise ValueError(f"TIFF image {tif_paths[i]} only contains one channel")
        return img

    return MosaicSource(
        name=tiff_folder.name,
        layout=MosaicLayout.from_paths(tif_paths),
        read_image=read_image,
    )


def _open_cvw_file(
    cvw_path: Path, convert_first_n_events: Optional[int]
) -> MosaicSource:
    """Images reconstructed from a `.cvw` file, one tile per event."""
    cvw = CvwFile(cvw_path)
    # Image sizes come from the file index, without reconstructing any image
    n_rows = cvw.n_rows[:convert_first_n_events]
    pixel_size_y, pixel_size_x = cvw.pixel_size
    layout = MosaicLayout(
        tile=TileSize(y=int(n_rows.max()), x=IMAGE_WIDTH),
        pixel_size=PixelSize(y=pixel_size_y, x=pixel_size_x),
        n_tiles=len(n_rows),
        channel_labels=tuple(sanitize_string(name) for name in cvw.channel_names),
    )
    return MosaicSource(name=cvw_path.stem, layout=layout, read_image=cvw.read_image)


def _write_mosaic(
    zarr_array: Array,
    source: MosaicSource,
    block_size: tuple[int, int],
    add_z_singleton: bool,
    convert_to_uint16: bool,
) -> list[Roi]:
    """Fill the mosaic block by block and return one ROI per tile, in tile order.

    Each block of whole tiles is assembled in memory, zero-padded, and written
    with a single assignment, so each chunk (or shard) is written only once.

    Args:
        zarr_array: Mosaic array, (c, y, x) or (c, z, y, x) with a singleton z.
        source: Images to place in the mosaic.
        block_size: (y, x) block size in pixels, a whole number of tiles.
        add_z_singleton: Whether the array has a singleton z axis.
        convert_to_uint16: Whether to convert each block with `float_to_uint16`
            before writing it.
    """
    layout = source.layout
    tile = layout.tile
    block_rows, block_cols = block_size[0] // tile.y, block_size[1] // tile.x
    rois: list[Roi] = []
    for row0, col0 in itertools.product(
        range(0, layout.n_rows, block_rows), range(0, layout.n_cols, block_cols)
    ):
        rows = range(row0, min(row0 + block_rows, layout.n_rows))
        cols = range(col0, min(col0 + block_cols, layout.n_cols))
        block = np.zeros(
            (layout.n_channels, len(rows) * tile.y, len(cols) * tile.x),
            dtype=np.float32,
        )
        for row, col in itertools.product(rows, cols):
            i = row * layout.n_cols + col
            if i >= layout.n_tiles:
                continue
            img = source.read_image(i)
            y0, x0 = (row - row0) * tile.y, (col - col0) * tile.x
            block[:, y0 : y0 + img.shape[1], x0 : x0 + img.shape[2]] = img
            roi_slices = layout.get_roi_slices(
                row=row, col=col, img_shape=img.shape[-2:]
            )
            rois.append(Roi(name=str(i), slices=roi_slices, space="world"))

        ys = slice(row0 * tile.y, row0 * tile.y + block.shape[1])
        xs = slice(col0 * tile.x, col0 * tile.x + block.shape[2])
        if convert_to_uint16:
            block = float_to_uint16(block)
        if add_z_singleton:
            zarr_array[:, 0, ys, xs] = block
        else:
            zarr_array[:, ys, xs] = block

    return sorted(rois, key=lambda roi: int(roi.name))


def read_fcs_events(fcs_path: Path) -> FeatureTable:
    """Read all events of an FCS file into a feature table indexed by event_index.

    Columns are the raw ($PnN) parameter names, unchanged. Values are read as
    float32, even for parameters stored as integers. The event_index is the
    position of the event in the file, which matches the FOV ROI names.
    """
    fcs = flowio.FlowData(fcs_path)
    events = fcs.as_array(preprocess=False).astype(np.float32)
    data = pl.DataFrame(events, schema=fcs.pnn_labels, orient="row")
    data = data.with_columns(pl.Series(EVENT_INDEX_KEY, np.arange(len(data))))
    return FeatureTable(data, meta=FeatureTableMeta(index_key=EVENT_INDEX_KEY))


def write_fcs_metadata(fcs_path: Path, metadata_path: Path) -> None:
    """Dump all TEXT keywords of an FCS file to a JSON file under "fcs_text".

    Keywords are as read by flowio (lowercased, without the leading "$"),
    values are the unchanged strings. An existing file is overwritten.
    """
    fcs = flowio.FlowData(fcs_path, only_text=True)
    with open(metadata_path, "w") as f:
        json.dump({"fcs_text": fcs.text}, f, indent=2)


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=fcf_bd_converter)
