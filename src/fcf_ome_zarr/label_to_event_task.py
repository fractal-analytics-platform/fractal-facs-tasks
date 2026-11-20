"""Map the labels of an existing label image to the FCS event of their FOV."""

import logging

import numpy as np
import polars as pl
from ngio import OmeZarrContainer, open_ome_zarr_container
from ngio.tables import FeatureTable
from pydantic import validate_call

from fcf_ome_zarr.constants import EVENT_INDEX_KEY, FOV_ROI_TABLE, TABLE_BACKEND

logger = logging.getLogger("label_to_event_task")


def label_to_event_table_name(label_name: str) -> str:
    """Name of the table linking the labels of label_name to their FCS events."""
    return f"{label_name}_to_event"


def write_label_to_event_table(
    ome_zarr: OmeZarrContainer,
    label_name: str,
    labels: list[int],
    event_indices: list[int],
    overwrite: bool,
) -> None:
    """Write the "<label_name>_to_event" table, mapping each label to its event."""
    label_to_event = pl.DataFrame({"label": labels, EVENT_INDEX_KEY: event_indices})
    ome_zarr.add_table(
        name=label_to_event_table_name(label_name),
        table=FeatureTable(table_data=label_to_event, reference_label=label_name),
        overwrite=overwrite,
        backend=TABLE_BACKEND,
    )


@validate_call
def label_to_event_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    label_name: str,
    overwrite: bool = True,
) -> None:
    """Link the labels of an existing label image to their FCS events.

    Writes the "<label_name>_to_event" feature table, with the FCS event of
    each label in the "event_index" column.
    Each label must lie entirely within a single FOV of the FOV_ROI_table.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        label_name (str): Name of the existing label image.
        overwrite (bool): Whether to overwrite an existing label to event
            table. Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    table_name = label_to_event_table_name(label_name)
    if not overwrite and table_name in ome_zarr.list_tables():
        raise FileExistsError(
            f"Table {table_name} already exists. Set overwrite=True to overwrite it."
        )

    label_ngio = ome_zarr.get_label(label_name)
    logger.info(f"{label_ngio=}")
    fov_rois = ome_zarr.get_generic_roi_table(name=FOV_ROI_TABLE).rois()

    labels: list[int] = []
    event_indices: list[int] = []
    n_pixels_in_fovs = 0
    for fov_roi in fov_rois:
        fov_label = label_ngio.get_roi_as_numpy(fov_roi)
        fov_labels = np.unique(fov_label[fov_label > 0]).tolist()
        labels.extend(fov_labels)
        # FOV names are the FCS event index
        event_indices.extend([int(fov_roi.get_name())] * len(fov_labels))
        n_pixels_in_fovs += int(np.count_nonzero(fov_label))

    # FOVs don't overlap, so these two checks ensure each label lies in one FOV
    if len(labels) != len(set(labels)):
        raise ValueError(f"Some labels of {label_name} are found in more than one FOV.")
    n_pixels = int(label_ngio.get_as_dask().astype(bool).sum().compute())
    if n_pixels != n_pixels_in_fovs:
        raise ValueError(
            f"{n_pixels - n_pixels_in_fovs} labeled pixels of {label_name} "
            "lie outside the FOVs."
        )

    write_label_to_event_table(
        ome_zarr, label_name, labels, event_indices, overwrite=overwrite
    )
    logger.info(f"Table {table_name} successfully created at {zarr_url}")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=label_to_event_task)
