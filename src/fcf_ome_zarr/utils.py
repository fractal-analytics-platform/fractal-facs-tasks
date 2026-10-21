"""Shared helpers: pydantic models for iterator configuration and string utils."""

import re
from typing import Literal

from pydantic import BaseModel, Field


def sanitize_string(value: str) -> str:
    """Turn a string into a name that is safe to use in OME-Zarrs and as a parameter.

    Used e.g. for OME-Zarr names and channel labels. Only letters, digits,
    underscores and dashes are kept. Every run of other characters is replaced
    by a single underscore, and underscores at both ends are removed.
    See https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/843
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


class MaskingConfiguration(BaseModel):
    """Masking configuration."""

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    """
    Mode of masking to be applied. If "Table Name", the identifier refers to a
    masking table name.
    If "Table Name", the identifier refers to a masking table name.
    If "Label Name", the identifier refers to a label image name.
    """
    identifier: str | None = None
    """
    Name of the masking table or label image depending on the mode.
    """


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration."""

    masking: MaskingConfiguration | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    """
    If configured, the segmentation will be only saved within the mask region.
    will be applied based on the provided masking configuration.
    """
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
    """
    Name of a ROI table. If provided, the segmentation will be performed for each ROI
    in the specified ROI table.
    """
