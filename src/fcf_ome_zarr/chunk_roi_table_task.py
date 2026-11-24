"""Build an ROI table indexing whole chunks (or shards) of the image."""

import itertools
import logging

from ngio import Roi, open_ome_zarr_container
from ngio.tables import RoiTable
from pydantic import PositiveInt, validate_call

from fcf_ome_zarr.constants import TABLE_BACKEND

logger = logging.getLogger("chunk_roi_table_task")


@validate_call
def chunk_roi_table_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    chunks_per_roi_side: PositiveInt = 1,
    table_name: str | None = None,
    overwrite: bool = True,
) -> None:
    """Build an ROI table whose ROIs cover whole chunks (or shards) of the image.

    Useful to run tasks that iterate over ROI tables chunk by chunk. If the
    image is sharded, ROIs cover whole shards, otherwise whole chunks. Only the
    y and x axes are split; ROIs span all other axes. ROIs at the image edge are
    clipped to the image. Only the highest resolution level is considered.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        chunks_per_roi_side (int): Number of chunks (or shards) per ROI side
            along y and x, e.g. 2 means each ROI covers a 2 x 2 grid of chunks.
            Defaults to 1.
        table_name (str | None): Name of the ROI table. If None, it is
            "chunk_ROI_table_{n}x{n}", or "shard_ROI_table_{n}x{n}" if the image
            is sharded, with n = chunks_per_roi_side. Defaults to None.
        overwrite (bool): Whether to overwrite an existing table. Defaults to
            True.
    """
    logger.info(f"{zarr_url=}")

    ome_zarr = open_ome_zarr_container(zarr_url)
    # Highest resolution level; only metadata is read, no pixel data
    image = ome_zarr.get_image()

    if table_name is None:
        block_type = "chunk" if image.zarr_array.shards is None else "shard"
        n = chunks_per_roi_side
        table_name = f"{block_type}_ROI_table_{n}x{n}"
    if not overwrite and table_name in ome_zarr.list_tables():
        raise FileExistsError(
            f"Table {table_name} already exists. Set overwrite=True to overwrite it."
        )

    # Shard shape if the array is sharded, chunk shape otherwise
    block = dict(zip(image.axes, image.write_granularity, strict=True))
    logger.info(f"Chunk/shard shape: {block}")
    step_y = block["y"] * chunks_per_roi_side
    step_x = block["x"] * chunks_per_roi_side
    size_y = image.dimensions.get("y", default=None)
    size_x = image.dimensions.get("x", default=None)
    if size_y is None or size_x is None:
        raise ValueError(f"Image at {zarr_url} has no y or x axis.")

    # Full image slices in pixels, so that ROIs span all non-(y, x) axes
    image_slices = {
        s.axis_name: s for s in image.roi().to_pixel(image.pixel_size).slices
    }
    rois = []
    for i, (y0, x0) in enumerate(
        itertools.product(range(0, size_y, step_y), range(0, size_x, step_x))
    ):
        slices = image_slices | {
            "y": slice(y0, min(y0 + step_y, size_y)),
            "x": slice(x0, min(x0 + step_x, size_x)),
        }
        roi = Roi.from_values(slices=slices, name=str(i), space="pixel")
        rois.append(roi.to_world(image.pixel_size))

    ome_zarr.add_table(
        name=table_name,
        table=RoiTable(rois=rois),
        overwrite=overwrite,
        backend=TABLE_BACKEND,
    )
    logger.info(f"Table {table_name} with {len(rois)} ROIs created at {zarr_url}")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=chunk_roi_table_task)
