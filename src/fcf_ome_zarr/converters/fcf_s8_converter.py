"""Converter for the BD S8 cell sorter imaging data at UZH FCF"""

import logging
from pathlib import Path
from typing import Optional

import ngio
import numpy as np
import tifffile
from ngio import Roi
from ngio.tables import RoiTable
from pydantic import validate_call
from zarr.storage import LocalStore

from fcf_ome_zarr.converters.mosaic import MosaicLayout, pad_to_shape


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
        channels_meta=list(layout.channel_labels),
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
        padded = pad_to_shape(img, target_shape=layout.tile)
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
    # Set contrast limits for viewers
    ome_zarr_container.set_channel_windows_with_percentiles(percentiles=(0.05, 99.995))

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
