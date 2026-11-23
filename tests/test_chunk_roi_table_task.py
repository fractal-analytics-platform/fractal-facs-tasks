from pathlib import Path

import pytest
from ngio import open_ome_zarr_container

from fcf_ome_zarr.chunk_roi_table_task import chunk_roi_table_task
from fcf_ome_zarr.converters.fcf_bd_converter import fcf_bd_converter


def _convert(tmp_path: Path, tiff_dir: Path, **kwargs) -> str:
    result = fcf_bd_converter(
        tiff_folder_path=str(tiff_dir),
        zarr_dir=str(tmp_path),
        fake_a_plate=False,
        **kwargs,
    )
    assert result is not None
    return result["image_list_updates"][0]["zarr_url"]


def _roi_boxes_px(zarr_url: str) -> list[tuple[int, int, int, int]]:
    """(y_start, y_end, x_start, x_end) in pixels of each ROI of the table."""
    ome_zarr = open_ome_zarr_container(zarr_url)
    pixel_size = ome_zarr.get_image().pixel_size
    boxes = []
    for roi in ome_zarr.get_roi_table("chunk_ROI_table").rois():
        slices = roi.to_slicing_dict(pixel_size)
        y, x = slices["y"], slices["x"]
        boxes.append((round(y.start), round(y.stop), round(x.start), round(x.stop)))
    return boxes


# The S8 mosaic is 146 x 208 px, made of 2 x 2 tiles of 73 x 104 px
@pytest.mark.parametrize(
    ("converter_kwargs", "chunks_per_roi_side", "expected"),
    [
        # Chunks of one tile
        (
            {"target_chunk_size": 1},
            1,
            [
                (0, 73, 0, 104),
                (0, 73, 104, 208),
                (73, 146, 0, 104),
                (73, 146, 104, 208),
            ],
        ),
        # Shards of 2 x 1 tiles are used instead of the one tile chunks
        (
            {"use_sharding": True, "target_shard_size": 150},
            1,
            [(0, 146, 0, 104), (0, 146, 104, 208)],
        ),
        # 2 x 2 shards, clipped to the image
        (
            {"use_sharding": True, "target_shard_size": 150},
            2,
            [(0, 146, 0, 208)],
        ),
    ],
)
def test_chunk_roi_table(
    tmp_path: Path,
    bd_s8_tiff_dir: Path,
    converter_kwargs: dict,
    chunks_per_roi_side: int,
    expected: list[tuple[int, int, int, int]],
):
    zarr_url = _convert(tmp_path, bd_s8_tiff_dir, **converter_kwargs)
    chunk_roi_table_task(zarr_url=zarr_url, chunks_per_roi_side=chunks_per_roi_side)
    assert _roi_boxes_px(zarr_url) == expected
