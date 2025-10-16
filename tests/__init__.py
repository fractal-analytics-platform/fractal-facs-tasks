import json
from pathlib import Path

import fcf_ome_zarr

PACKAGE_DIR = Path(fcf_ome_zarr.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
