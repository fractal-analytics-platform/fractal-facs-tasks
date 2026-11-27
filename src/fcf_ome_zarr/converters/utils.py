"""Input parameter models of the BD converter."""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, PositiveInt


class Chunks(BaseModel):
    """Store the image in chunks, without sharding."""

    mode: Literal["Chunks"] = "Chunks"
    """Store the image in chunks, without sharding."""
    target_chunk_size: int = 512
    """
    Target size in pixels of the chunks along y and x. Rounded to the closest whole
    number of tiles (at least one).
    """


class Shards(BaseModel):
    """Store the image in shards, with one mosaic tile per chunk."""

    mode: Literal["Shards"] = "Shards"
    """Store the image in shards, with one mosaic tile per chunk."""
    target_shard_size: int = 2048
    """
    Target size in pixels of the shards along y and x. Rounded to the closest whole
    number of tiles (at least one).
    """


AnyStorage = Annotated[Chunks | Shards, Field(discriminator="mode")]


class AdvancedBDConverterParameters(BaseModel):
    """Advanced parameters of the BD converter."""

    convert_to_uint16: bool = True
    """
    Whether to store the image as uint16 instead of float32 (original data type).
    Values are clipped to [0, 8] and scaled linearly to the full uint16 range.
    We recommend to keep this option selected, since many processing tasks and
    display software may not correctly support the original data type and range.
    """
    fake_a_plate: bool = False
    """
    Whether to create a fake plate structure with a single well "A01" containing the
    mosaic image. This is useful to use some tools that need plates (e.g. Fractal
    feature explorer).
    """
    add_z_singleton: bool = False
    """
    Whether to add a singleton Z dimension to the OME-Zarr. This is useful to use some
    of the tools (like the Fractal feature explorer) that require Z singleton
    dimensions.
    """
    storage: AnyStorage = Field(default_factory=Chunks)
    """
    How to store the image: in chunks without sharding, or in shards with one mosaic
    tile per chunk. Chunks and shards always hold all channels and a whole number of
    tiles.
    """
    convert_first_n_events: Optional[PositiveInt] = None
    """
    If provided, only the first N events are converted: the first N TIFF files of the
    folder, or the first N events of the `.cvw` file. This is useful for testing with
    fewer events. If None, all events are converted.
    """
