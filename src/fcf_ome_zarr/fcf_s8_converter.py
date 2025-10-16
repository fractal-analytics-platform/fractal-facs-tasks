"""This is the Python module for my_task."""

import logging
import math
import random
from pathlib import Path
from typing import Optional

import ngio
import numpy as np
import tifffile
import zarr
from ngio import Roi
from ngio.tables import RoiTable
from pydantic import validate_call


def find_max_dimensions(tif_folder_path):
    """Find the maximum dimensions (y, x) across all TIFF files in the folder."""
    max_y = 0
    max_x = 0
    for tif_path in tif_folder_path.glob("*.tif*"):
        with tifffile.TiffFile(tif_path) as tif:
            shape = tif.pages[0].shape  # (channels, y, x) or (y, x, channels)
            if len(shape) == 3:
                # Assuming (channels, y, x)
                _, y, x = shape
            else:
                y, x = shape
            max_y = max(max_y, y)
            max_x = max(max_x, x)

    return max_y, max_x


def pad_to_shape(img, target_shape, dtype=np.float32):
    """Pad a 3D image (c, y, x) to the target shape (y, x) with zeros."""
    c, y, x = img.shape
    padded = np.zeros((c, *target_shape), dtype=dtype)
    padded[:, :y, :x] = img
    return padded


def sampled_percentiles(ac, low=0.5, high=99.5, n_chunks=200):
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


def reset_omero_channels(store, level="0"):
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
    xy_pixelsize: float = 1.0,
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
        xy_pixelsize: Pixel size in the XY dimensions. If possible, we should
            extract this from the metadata instead.
        overwrite (bool): Whether to overwrite an existing OME-Zarr image.
            Defaults to True.
    """
    logging.info(f"Processing {tiff_folder_path=}")
    tiff_folder_path = Path(tiff_folder_path)

    plate_url = "plate.ome.zarr"
    image_name = "s8_cells_mosaic.ome.zarr"
    if fake_a_plate:
        zarr_url = Path(f"{zarr_dir}/{plate_url}/{image_name}")

        plate = ngio.create_empty_plate(plate_url, name=plate_url, overwrite=True)
        well = plate.add_well(row="A", column=1)
        well.add_image(image_name)

    else:
        zarr_url = Path(f"{zarr_dir}/{image_name}")

    # Convert the TIFFs to a mosaic OME-Zarr
    if convert_first_x_tiffs is not None:
        tif_paths = sorted(tiff_folder_path.glob("*.tif*"))[:convert_first_x_tiffs]
    else:
        tif_paths = sorted(tiff_folder_path.glob("*.tif*"))

    n_images = len(tif_paths)
    n_channels = 6
    max_y, max_x = find_max_dimensions(tiff_folder_path)

    # Create a mosaci grid & make it as square as possible
    n_cols = math.ceil(math.sqrt(n_images * max_y / max_x))
    n_rows = math.ceil(n_images / n_cols)

    total_y = n_rows * max_y
    total_x = n_cols * max_x
    logging.info(
        f"Grid layout: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} slots"
    )
    logging.info(f"Total image size: {total_y} y x {total_x} x")

    # Create OME-Zarr
    store = zarr.DirectoryStore(zarr_url)
    if add_z_singleton:
        ome_zarr_container = ngio.create_empty_ome_zarr(
            store,
            shape=(n_channels, 1, total_y, total_x),
            axes_names=["c", "z", "y", "x"],
            xy_pixelsize=xy_pixelsize,
            dtype="float32",
            chunks=(6, 1, 512, 512),  # adjust as needed
            overwrite=overwrite,
        )
    else:
        ome_zarr_container = ngio.create_empty_ome_zarr(
            store,
            shape=(n_channels, total_y, total_x),
            axes_names=["c", "y", "x"],
            xy_pixelsize=xy_pixelsize,
            dtype="float32",
            chunks=(6, 512, 512),  # adjust as needed
            overwrite=overwrite,
        )
    ngio_image = ome_zarr_container.get_image()
    za = ngio_image.zarr_array

    rois = []

    # Loop over TIFFs & place them inside the mosaic
    for i, tif_path in enumerate(tif_paths):
        row, col = divmod(i, n_cols)
        y0, x0 = row * max_y, col * max_x

        with tifffile.TiffFile(tif_path) as tif:
            img = tif.asarray().astype(np.float32)
            y_true, x_true = img.shape[-2:]
            if img.shape[0] != n_channels and img.ndim == 2:
                img = img[None, :, :]  # add channel if missing
        padded = pad_to_shape(img, (max_y, max_x))

        if add_z_singleton:
            za[:, 0, y0 : y0 + max_y, x0 : x0 + max_x] = padded

        else:
            za[:, y0 : y0 + max_y, x0 : x0 + max_x] = padded

        rois.append(
            Roi(
                name=str(i),
                x=x0 * xy_pixelsize,
                y=y0 * xy_pixelsize,
                x_length=x_true * xy_pixelsize,
                y_length=y_true * xy_pixelsize,
            )
        )

    # --- Create ROI table ---
    roi_table = RoiTable(rois=rois)
    ome_zarr_container.add_table("FOV_ROI_table", roi_table, overwrite=overwrite)
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
