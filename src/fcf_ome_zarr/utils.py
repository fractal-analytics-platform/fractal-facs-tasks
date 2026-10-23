"""Shared helpers: string utils."""

import re


def sanitize_string(value: str) -> str:
    """Turn a string into a name that is safe to use in OME-Zarrs and as a parameter.

    Used e.g. for OME-Zarr names and channel labels. Only letters, digits,
    underscores and dashes are kept. Every run of other characters is replaced
    by a single underscore, and underscores at both ends are removed.
    See https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/843
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
