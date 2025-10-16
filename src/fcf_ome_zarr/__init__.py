"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fcf_ome_zarr")
except PackageNotFoundError:
    __version__ = "uninstalled"
