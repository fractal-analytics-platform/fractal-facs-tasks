from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from fcf_ome_zarr.converters.mosaic import (
    MosaicLayout,
    PixelSize,
    TileSize,
    pad_to_shape,
)


def make_layout(
    n_tiles: int,
    tile: tuple[int, int] = (10, 10),
    pixel_size: tuple[float, float] = (1.0, 1.0),
    n_channels: int = 1,
) -> MosaicLayout:
    return MosaicLayout(
        tile=TileSize(*tile),
        pixel_size=PixelSize(*pixel_size),
        n_tiles=n_tiles,
        channel_labels=tuple(f"channel_{i}" for i in range(n_channels)),
    )


@pytest.mark.parametrize(
    "n_tiles, tile, n_rows, n_cols",
    [
        (1, (10, 10), 1, 1),
        (16, (10, 10), 4, 4),
        (7, (10, 10), 3, 3),
        (16, (10, 40), 8, 2),
        (16, (40, 10), 2, 8),
    ],
)
def test_grid_shape_balances_mosaic_aspect(
    n_tiles: int, tile: tuple[int, int], n_rows: int, n_cols: int
):
    layout = make_layout(n_tiles=n_tiles, tile=tile)
    assert (layout.n_rows, layout.n_cols) == (n_rows, n_cols)


@pytest.mark.parametrize("tile", [(10, 10), (10, 40), (40, 10), (73, 104)])
def test_grid_fits_all_tiles_without_empty_row(tile: tuple[int, int]):
    for n_tiles in range(1, 30):
        layout = make_layout(n_tiles=n_tiles, tile=tile)
        assert layout.n_rows * layout.n_cols >= n_tiles, n_tiles
        assert (layout.n_rows - 1) * layout.n_cols < n_tiles, n_tiles


def test_size_is_grid_shape_times_tile_size():
    layout = make_layout(n_tiles=7, tile=(10, 20))
    assert layout.size == (layout.n_rows * 10, layout.n_cols * 20)


@pytest.mark.parametrize("add_z_singleton", [True, False])
def test_tile_slices_stay_in_bounds_without_overlap(add_z_singleton: bool):
    layout = make_layout(n_tiles=7, tile=(10, 20), n_channels=3)
    shape = (3, 1, *layout.size) if add_z_singleton else (3, *layout.size)
    hits = np.zeros(shape, dtype=int)

    for i in range(layout.n_tiles):
        row, col = divmod(i, layout.n_cols)
        tile_slice = layout.get_tile_slice(
            row=row, col=col, add_z_singleton=add_z_singleton
        )
        assert hits[tile_slice].shape == (3, 10, 20)
        hits[tile_slice] += 1

    assert hits.max() == 1
    assert hits.sum() == layout.n_tiles * 3 * 10 * 20


def test_from_paths_reads_tile_size_and_metadata(
    example_data_dir: Path, example_channel_labels: tuple[str, ...]
):
    tif_paths = sorted(example_data_dir.glob("*.tif*"))
    layout = MosaicLayout.from_paths(tif_paths)

    # NOTE: this is hardcoding params from the example data.
    # If the data is changed this should be updated.
    assert layout.tile == (73, 104)
    assert layout.n_tiles == 4
    assert layout.n_channels == 6
    assert layout.channel_labels == example_channel_labels
    assert layout.pixel_size == pytest.approx((113.636, 136.364), abs=1e-3)


def test_from_paths_without_files_raises():
    with pytest.raises(ValueError, match="No TIFF files"):
        MosaicLayout.from_paths([])


def test_from_paths_with_mixed_folder_raises(mixed_tiff_dir: Path):
    tif_paths = sorted(mixed_tiff_dir.glob("*.tif*"))
    with pytest.raises(ValueError, match="mismatching channels"):
        MosaicLayout.from_paths(tif_paths)


def test_from_paths_uses_only_given_files(example_data_dir: Path):
    tif_paths = sorted(example_data_dir.glob("*.tif*"))
    layout = MosaicLayout.from_paths(tif_paths[:1])

    assert layout.n_tiles == 1
    assert layout.tile == (41, 104)


def test_from_paths_with_mismatching_pixel_size_raises(
    make_tiff: Callable[..., Path],
):
    tif_paths = [
        make_tiff("a.tif", pixel_size=(2.0, 4.0)),
        make_tiff("b.tif", pixel_size=(2.0, 5.0)),
    ]
    with pytest.raises(ValueError, match="mismatching pixel size"):
        MosaicLayout.from_paths(tif_paths)


def test_from_paths_without_page_names_uses_default_labels(
    make_tiff: Callable[..., Path],
):
    tif_paths = [make_tiff("a.tif", shape=(3, 20, 30))]
    layout = MosaicLayout.from_paths(tif_paths)

    assert layout.channel_labels == ("channel_0", "channel_1", "channel_2")


def test_pad_to_shape_keeps_image_in_top_left_corner():
    img = np.ones((2, 3, 4), dtype=np.float32)
    padded = pad_to_shape(img, target_shape=(5, 6))

    assert padded.shape == (2, 5, 6)
    assert np.array_equal(padded[:, :3, :4], img)
    assert padded.sum() == img.sum()


def test_pad_to_shape_with_image_larger_than_target_raises():
    img = np.ones((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        pad_to_shape(img, target_shape=(2, 2))
