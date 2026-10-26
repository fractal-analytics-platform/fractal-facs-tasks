"""Write the small test FCS file from the full BD S8 FCS export.

The full export (~17 MB, 10000 events x 418 parameters) is too big to commit.
This keeps only the events matching the TIFFs in Demo_data_S8 and a few
parameters. The TEXT metadata is not kept, flowio writes a minimal one.
The raw values are written as float32, so they are bit-identical to the source.

Usage:
    python make_test_fcs.py <full_fcs_path> <output_fcs_path>
"""

import sys
from pathlib import Path

import flowio
import numpy as np

N_EVENTS = 4
COLUMNS = [
    "FSC-A",
    "FSC-H",
    "SSC (Imaging)-A",
    "LightLoss (Imaging)-A",
    "Size (FSC)",
    "Total Intensity (FSC)",
    "Time",
    "SpectralEventWidth",
]


def main(full_fcs_path: Path, output_fcs_path: Path) -> None:
    """Write the first N_EVENTS events and COLUMNS of the full FCS file."""
    fcs = flowio.FlowData(full_fcs_path)
    col_idx = [fcs.pnn_labels.index(name) for name in COLUMNS]
    events = fcs.as_array(preprocess=False)[:N_EVENTS, col_idx].astype(np.float32)
    with open(output_fcs_path, "wb") as fh:
        flowio.create_fcs(fh, events.ravel(), COLUMNS)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
