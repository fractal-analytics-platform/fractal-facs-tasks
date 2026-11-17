"""Reader for BD `.cvw` waveform files.

A `.cvw` file stores one raw waveform record per event, followed by an index
of the records and a JSON metadata block. All values are little-endian.

Layout:
    - Header: `b"CHORWAVE"` magic, number of columns, number of events and the
      offset of the JSON block.
    - Records: per event, `T x n_columns` interleaved uint16 samples followed by
      a 4096-byte footer.
    - Index: the `24 * n_events` bytes just before the JSON. Per event, three
      uint64 values: event number, record offset, record length.
    - JSON: from the JSON offset to the end of the file.
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Final

import numpy as np

# Header
CVW_MAGIC: Final = b"CHORWAVE"
HEADER_N_COLUMNS_OFFSET: Final = 0x0C
HEADER_N_EVENTS_OFFSET: Final = 0x3C
HEADER_JSON_OFFSET: Final = 0x50

# Index: one (event_number, offset, length) uint64 triplet per event
INDEX_DTYPE: Final = np.dtype(
    [("event_number", "<u8"), ("offset", "<u8"), ("length", "<u8")]
)

# Record footer, read as uint32 values
FOOTER_BYTES: Final = 4096
FOOTER_MAGIC: Final = 0xBDBDBDBD
FOOTER_EVENT_NUMBER_IDX: Final = 4

# Image geometry: one image row per sliding window over the waveform
WINDOW_SAMPLES: Final = 256
WINDOW_HOP: Final = 64
IMAGE_WIDTH: Final = 104

# Image channels are JSON keys "2"-"7". With 8 columns the column equals the key,
# with 6 columns the column is the key minus 2.
IMAGE_CHANNEL_KEYS: Final = ("2", "3", "4", "5", "6", "7")
COLUMN_OFFSET: Final = {8: 0, 6: 2}

SAMPLE_DTYPE: Final = np.dtype("<u2")


class CvwFile:
    """Random-access reader for a BD `.cvw` waveform file.

    The file is memory-mapped; records are only read when requested.

    Attributes:
        path: Path to the `.cvw` file.
        n_columns: Number of interleaved waveform columns (8 or 6).
        n_events: Number of event records.
        metadata: Parsed JSON metadata block.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a `.cvw` file and parse its header, index and JSON metadata.

        Args:
            path: Path to the `.cvw` file.

        Raises:
            ValueError: If the file does not start with the `.cvw` magic.
        """
        self.path = Path(path)
        self._data = np.memmap(self.path, dtype=np.uint8, mode="r")

        magic = self._data[: len(CVW_MAGIC)].tobytes()
        if magic != CVW_MAGIC:
            raise ValueError(
                f"{self.path}: bad magic {magic!r}, expected {CVW_MAGIC!r}"
            )

        self.n_columns = int(self._read_scalar(HEADER_N_COLUMNS_OFFSET, "<u4"))
        self.n_events = int(self._read_scalar(HEADER_N_EVENTS_OFFSET, "<u4"))
        json_offset = int(self._read_scalar(HEADER_JSON_OFFSET, "<u8"))

        index_offset = json_offset - INDEX_DTYPE.itemsize * self.n_events
        self._index = self._data[index_offset:json_offset].view(INDEX_DTYPE)
        self.metadata: dict[str, Any] = json.loads(
            self._data[json_offset:].tobytes().decode().strip()
        )

    def _read_scalar(self, offset: int, dtype: str) -> np.generic:
        """Read a single little-endian scalar from the file."""
        size = np.dtype(dtype).itemsize
        return self._data[offset : offset + size].view(dtype)[0]

    @property
    def event_numbers(self) -> np.ndarray:
        """Event number of each record."""
        return self._index["event_number"]

    @cached_property
    def n_samples(self) -> np.ndarray:
        """Number of samples `T` per column, for each record."""
        sample_bytes = self._index["length"] - FOOTER_BYTES
        return (sample_bytes // (SAMPLE_DTYPE.itemsize * self.n_columns)).astype(
            np.int64
        )

    @cached_property
    def n_rows(self) -> np.ndarray:
        """Number of image rows `J` for each record."""
        return (self.n_samples - WINDOW_SAMPLES) // WINDOW_HOP + 1

    def image_shape(self, i: int) -> tuple[int, int]:
        """(rows, width) of the image of event `i`, from the index alone."""
        return int(self.n_rows[i]), IMAGE_WIDTH

    @cached_property
    def channel_names(self) -> tuple[str, ...]:
        """Names of the image channels, in channel order "2"-"7"."""
        names = self.metadata["namesForImageChannels"]
        return tuple(names[k] for k in IMAGE_CHANNEL_KEYS)

    @cached_property
    def channel_columns(self) -> tuple[int, ...]:
        """Waveform column of each image channel, in channel order "2"-"7"."""
        shift = COLUMN_OFFSET[self.n_columns]
        return tuple(int(k) - shift for k in IMAGE_CHANNEL_KEYS)

    def raw_waveform(self, i: int) -> np.ndarray:
        """Raw waveform of event `i`.

        The footer magic and event number are checked to catch misaligned reads.

        Args:
            i: Event index, i.e. the position of the record in the file.

        Returns:
            Read-only `(T, n_columns)` uint16 view into the memory-mapped file.

        Raises:
            ValueError: If the record footer does not match the index.
        """
        offset = int(self._index["offset"][i])
        sample_end = offset + int(self._index["length"][i]) - FOOTER_BYTES

        footer = self._data[sample_end : sample_end + FOOTER_BYTES].view("<u4")
        event_number = int(self.event_numbers[i])
        if footer[0] != FOOTER_MAGIC or footer[FOOTER_EVENT_NUMBER_IDX] != event_number:
            raise ValueError(
                f"{self.path}: event {i} footer does not match the index "
                f"(magic {int(footer[0]):#x}, "
                f"event number {int(footer[FOOTER_EVENT_NUMBER_IDX])} "
                f"vs {event_number})"
            )

        samples = self._data[offset:sample_end].view(SAMPLE_DTYPE)
        return samples.reshape(-1, self.n_columns)
