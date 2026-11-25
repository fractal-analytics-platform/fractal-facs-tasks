"""Invert one channel of the mosaic image and add it as a new channel."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from ngio import open_ome_zarr_container
from numpy.typing import NDArray
from pydantic import validate_call

from fcf_ome_zarr.constants import FCS_METADATA_FILE, FOV_ROI_TABLE

logger = logging.getLogger("invert_channel_task")


def invert(img: NDArray[Any], background: float) -> NDArray[Any]:
    """Invert an image around its background: max(0, background - x).

    The result has the dtype of the input; integers are rounded.
    """
    inverted = np.clip(background - img.astype(np.float64), 0, None)
    if np.issubdtype(img.dtype, np.integer):
        inverted = np.round(inverted)
    return inverted.astype(img.dtype)


@validate_call
def invert_channel_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Input parameters
    channel_label: str,
) -> None:
    """Invert one channel and add it to the image as "<channel_label>_inv".

    The channel is inverted around its background c as max(0, c - x), so that
    objects darker than the background become bright. c is the median of the
    per-FOV medians of the channel, over the ROIs of the FOV_ROI_table, in the
    units of the image. It is logged. The inverted channel is written only
    inside the FOVs, so the padding between FOVs stays 0. The window of the new
    channel is the source window inverted the same way.

    The image is rewritten with the extra channel next to the input, then
    swapped in place of it. Labels, tables and the FCS metadata are kept.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channel_label (str): Label of the channel to invert.
    """
    logger.info(f"{zarr_url=}, {channel_label=}")

    ome_zarr = open_ome_zarr_container(zarr_url)
    image = ome_zarr.get_image()
    channel_labels = image.channel_labels
    inv_label = f"{channel_label}_inv"
    if channel_label not in channel_labels:
        raise ValueError(f"Channel {channel_label!r} not found in {channel_labels}.")
    if inv_label in channel_labels:
        raise ValueError(f"Channel {inv_label!r} already exists in {zarr_url}.")

    # Unsigned integer (uint16) or float (float32), as written by the converter
    if np.dtype(image.dtype).kind not in "uf":
        raise TypeError(f"Cannot invert images of dtype {image.dtype}.")

    # First pass: background c, leaving out the padding between FOVs
    rois = ome_zarr.get_roi_table(FOV_ROI_TABLE).rois()
    fov_medians = [
        np.median(image.get_roi_as_numpy(roi, channel_selection=channel_label))
        for roi in rois
    ]
    background = float(np.median(fov_medians))
    logger.info(f"Background of {channel_label}: c = {background}")

    derived_url = f"{zarr_url}_inverted"
    if Path(derived_url).exists():
        raise FileExistsError(
            f"{derived_url} already exists, likely from a failed run. Delete it "
            "and run the task again."
        )

    # Chunks and shards hold all channels, as written by the converter
    n_channels = len(channel_labels) + 1
    chunks = tuple(
        n_channels if ax == "c" else n
        for ax, n in zip(image.axes, image.chunks, strict=True)
    )
    shards = None
    if image.zarr_array.shards is not None:
        shards = tuple(
            n_channels if ax == "c" else n
            for ax, n in zip(image.axes, image.zarr_array.shards, strict=True)
        )

    derived = ome_zarr.derive_image(
        store=derived_url,
        channels_policy=n_channels,
        channels_meta=[*image.channels_meta.channels, inv_label],
        chunks=chunks,
        shards=shards,
        copy_labels=True,
        copy_tables=True,
    )
    derived_image = derived.get_image()
    logger.info(f"Derived image: {derived_image=}")

    # Second pass: write only inside the FOVs, so the padding stays at 0
    for roi in rois:
        derived_image.set_roi(
            roi, image.get_roi_as_numpy(roi), channel_selection=channel_labels
        )
        source_img = image.get_roi_as_numpy(roi, channel_selection=channel_label)
        derived_image.set_roi(
            roi, invert(source_img, background), channel_selection=inv_label
        )
    derived_image.consolidate(mode="auto")

    # Keep the windows of the original channels; the new one mirrors the source.
    # Percentiles are not recomputed, since they would count the padding.
    windows = {
        ch.label: (ch.channel_visualisation.start, ch.channel_visualisation.end)
        for ch in image.channels_meta.channels
    }
    start, end = windows[channel_label]
    windows[inv_label] = (max(0.0, background - end), max(0.0, background - start))
    derived.set_channel_windows(
        starts_ends=[windows[label] for label in derived.channel_labels]
    )

    # Not managed by ngio, so copied by hand
    fcs_metadata = Path(zarr_url) / FCS_METADATA_FILE
    if fcs_metadata.exists():
        shutil.copy(fcs_metadata, Path(derived_url) / FCS_METADATA_FILE)

    # Swap the derived image in place of the input
    tmp_url = f"{zarr_url}_tmp"
    os.rename(zarr_url, tmp_url)
    os.rename(derived_url, zarr_url)
    shutil.rmtree(tmp_url)
    logger.info(f"Channel {inv_label} added to {zarr_url}")

    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=invert_channel_task)
