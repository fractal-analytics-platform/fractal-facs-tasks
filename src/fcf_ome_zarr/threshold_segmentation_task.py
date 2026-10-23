"""Threshold segmentation of cells in the light-loss channel of BD S8 FOVs."""

import logging
from typing import Any

import numpy as np
import scipy.ndimage as ndi
from ngio import ChannelSelectionModel, open_ome_zarr_container
from ngio.iterators import SegmentationIterator
from numpy.typing import NDArray
from pydantic import validate_call
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from skimage.measure import label
from skimage.morphology import remove_small_objects

logger = logging.getLogger("threshold_segmentation_task")

# Each FOV holds a single event, so segmentation always runs FOV by FOV
FOV_ROI_TABLE = "FOV_ROI_table"
DEFAULT_CHANNEL = ChannelSelectionModel(identifier="LightLoss_Imaging", mode="label")


def segment_cell(
    input_img: NDArray[Any],
    sigma: tuple[float, float] = (1.5, 1.0),
    max_size: int = 64,
) -> NDArray[np.uint32]:
    """Segment cells that appear darker than the background, as in the light-loss signal

    The image is contrast-stretched, smoothed with a Gaussian filter and
    thresholded with Yen's method. Holes are filled, small objects removed and
    the remaining objects labeled.

    Args:
        input_img (NDArray[Any]): 2D (y, x) input image.
        sigma (tuple[float, float]): Standard deviation of the Gaussian smoothing
            along (y, x).
        max_size (int): Objects with this many pixels or fewer are removed.

    Returns:
        label_img (NDArray[np.uint32]): Labeled image with the same shape as the
            input.
    """
    image = rescale_intensity(input_img)
    image_smooth = ndi.gaussian_filter(image, sigma=sigma)

    th = threshold_yen(image_smooth)
    binary_img = image_smooth < th
    binary_img = ndi.binary_fill_holes(binary_img)
    binary_img = remove_small_objects(binary_img, max_size=max_size)
    label_img = label(binary_img, connectivity=1)
    assert isinstance(label_img, np.ndarray)
    return label_img.astype(np.uint32)


@validate_call
def threshold_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channel: ChannelSelectionModel = DEFAULT_CHANNEL,
    label_name: str | None = None,
    sigma: tuple[float, float] = (1.5, 1.0),
    max_size: int = 64,
    overwrite: bool = True,
) -> None:
    """Segment cells using automatic thresholding.

    Cells are expected to be darker than the background (e.g. light-loss
    channel). Segmentation runs separately for each ROI of the FOV_ROI_table,
    and label are unique across the entire mosaic.
    Images are expected to be 2D, optionally with a singleton z axis, which is
    preserved in the output label.

    Args:
        zarr_url (str): URL to the OME-Zarr container
        channel (ChannelSelectionModel): Select the input channel to be used for
            segmentation. Defaults to the "LightLoss_Imaging" channel.
        label_name (str | None): Name of the resulting label image. If not provided,
            it will be set to "<channel_identifier>_thresholded".
        sigma (tuple[float, float]): Standard deviation of the Gaussian smoothing
            along (y, x).
        max_size (int): Objects with this many pixels or fewer are removed.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    # Use the first of input_paths
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    if label_name is None:
        label_name = f"{channel.identifier}_thresholded"
    label_ngio = ome_zarr.derive_label(name=label_name, overwrite=overwrite)
    logger.info(f"Output label image: {label_ngio=}")

    image_ngio = ome_zarr.get_image()
    logger.info(f"{image_ngio=}")
    iterator = SegmentationIterator(
        input_image=image_ngio,
        output_label=label_ngio,
        channel_selection=channel,
        axes_order="yx",
        consolidation_mode="auto",
    )
    # Iterate over 2D (y, x) planes; ngio maps them back onto the label axes,
    # so a singleton z axis is preserved
    iterator = iterator.by_yx()
    logger.info(f"Iterator created: {iterator=}")

    table = ome_zarr.get_generic_roi_table(name=FOV_ROI_TABLE)
    logger.info(f"ROI table retrieved: {table=}")
    iterator = iterator.product(table)
    logger.info(f"Iterator updated with ROI table: {iterator=}")

    # Keep track of the maximum label to ensure unique across iterations
    max_label: int = 0

    # Core processing loop
    logger.info("Starting processing...")
    for image_data, writer in iterator.iter_as_numpy():
        label_img = segment_cell(input_img=image_data, sigma=sigma, max_size=max_size)
        label_img = np.where(label_img == 0, 0, label_img + max_label)
        max_label = max(max_label, int(label_img.max()))
        writer(label_img)
    # No need to call label.consolidate()
    # SegmentationIterator handles this
    logger.info(f"label {label_name} successfully created at {zarr_url}")
    return None


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=threshold_segmentation_task)
