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

Images are reconstructed from the waveform with a sliding-window FFT, phase
alignment against the reference phasors, and calibration by the intrinsic,
compensation and lifetime phasors stored in the JSON block.
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

# Image geometry: one image row per sliding window over the waveform, one image
# column per FFT bin in [FFT_BIN_START, FFT_BIN_STOP)
WINDOW_SAMPLES: Final = 256
WINDOW_HOP: Final = 64
FFT_BIN_START: Final = 16
FFT_BIN_STOP: Final = 120
IMAGE_WIDTH: Final = FFT_BIN_STOP - FFT_BIN_START

# Width of the beam path spanned by the IMAGE_WIDTH columns, in µm
BEAM_WIDTH_UM: Final = 60.0

# Scale from FFT amplitude of uint16 samples to image intensity
INTENSITY_SCALE: Final = 65535 * 8

# Image channels are JSON keys "2"-"7". With 8 columns the column equals the key,
# with 6 columns the column is the key minus 2.
IMAGE_CHANNEL_KEYS: Final = ("2", "3", "4", "5", "6", "7")
COLUMN_OFFSET: Final = {8: 0, 6: 2}
# Channel used for phase alignment
PHASE_CHANNEL_KEY: Final = "2"
# Fluorescence channels, which can have compensation and lifetime phasors
FLUORESCENCE_CHANNEL_KEYS: Final = ("5", "6", "7")

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

    @cached_property
    def _laser(self) -> dict[str, Any]:
        """Calibration block of the (single) imaging laser."""
        return self.metadata["lasers"][0]

    @cached_property
    def pixel_size(self) -> tuple[float, float]:
        """(y, x) pixel size in µm.

        x is the beam width over the image width. The laser `pixelAspectRatio` is
        the x pixel size over the distance the object travels during one window
        of `WINDOW_SAMPLES`. Rows are `WINDOW_HOP` samples apart, so
        `y = x * WINDOW_HOP / (WINDOW_SAMPLES * pixelAspectRatio)`. The ratio
        depends on the flow velocity (nozzle), so it is read from each file.

        Raises:
            ValueError: If the laser metadata has no `pixelAspectRatio`.
        """
        aspect_ratio = self._laser.get("pixelAspectRatio")
        if aspect_ratio is None:
            raise ValueError(f"{self.path}: no pixelAspectRatio in the laser metadata")
        x = BEAM_WIDTH_UM / IMAGE_WIDTH
        y = x * WINDOW_HOP / (WINDOW_SAMPLES * aspect_ratio)
        return y, x

    @cached_property
    def ref_phasors(self) -> np.ndarray:
        """Reference phasors, `(WINDOW_SAMPLES // 2 + 1,)` complex."""
        return _to_complex(self._laser["refPhasors"])

    @cached_property
    def intrinsic_phasors(self) -> np.ndarray:
        """Intrinsic phasors per image channel, `(6, IMAGE_WIDTH)` complex."""
        phasors = self._laser["intrinsicPhasors"]
        return np.stack([_to_complex(phasors[k]) for k in IMAGE_CHANNEL_KEYS])

    @cached_property
    def lifetime_phasors(self) -> np.ndarray:
        """Lifetime phasors per image channel, `(6, IMAGE_WIDTH)` complex.

        Channels without lifetime phasors, or files without any, get ones.
        """
        phasors = self._laser.get("lifetimePhasors", {})
        return np.stack(
            [
                _to_complex(phasors[k]) if k in phasors else np.ones(IMAGE_WIDTH)
                for k in IMAGE_CHANNEL_KEYS
            ]
        )

    @cached_property
    def compensation_matrix(self) -> np.ndarray | None:
        """Compensation matrix over the fluorescence channels, if present.

        Element `[a, b]` is the weight of channel `b` in compensated channel `a`,
        both in `FLUORESCENCE_CHANNEL_KEYS` order.
        """
        matrix = self.metadata.get("imageCompensationMatrix")
        if matrix is None:
            return None
        return np.array(
            [
                [matrix[a][b] for b in FLUORESCENCE_CHANNEL_KEYS]
                for a in FLUORESCENCE_CHANNEL_KEYS
            ]
        )

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

    def read_image(self, i: int) -> np.ndarray:
        """Reconstruct the image of event `i` from its raw waveform.

        Args:
            i: Event index, i.e. the position of the record in the file.

        Returns:
            `(6, J, IMAGE_WIDTH)` float32 image, channels in `IMAGE_CHANNEL_KEYS`
            order.
        """
        waveform = self.raw_waveform(i)[:, self.channel_columns].astype(np.float64)

        # Sliding-window FFT: (J, 6, WINDOW_SAMPLES) windows, rectangular window
        windows = np.lib.stride_tricks.sliding_window_view(
            waveform, WINDOW_SAMPLES, axis=0
        )[::WINDOW_HOP]
        bins = np.arange(FFT_BIN_START, FFT_BIN_STOP)
        spectra = np.fft.rfft(windows, axis=-1)[..., bins]
        spectra = np.moveaxis(spectra, 0, 1)  # (6, J, IMAGE_WIDTH)

        # Undo the phase advance of each window relative to the first one
        row = np.arange(spectra.shape[1])[:, None]
        spectra *= np.exp(-2j * np.pi * bins * WINDOW_HOP * row / WINDOW_SAMPLES)

        # Phase alignment on the first row of the phase channel
        phase_idx = IMAGE_CHANNEL_KEYS.index(PHASE_CHANNEL_KEY)
        gain = spectra[phase_idx, 0] / self.ref_phasors[bins]
        phase = np.angle(np.sum(1 / gain))

        calibrated = spectra * np.exp(1j * phase) / INTENSITY_SCALE
        calibrated /= self.intrinsic_phasors[:, None, :]

        if self.compensation_matrix is not None:
            fluo = [IMAGE_CHANNEL_KEYS.index(k) for k in FLUORESCENCE_CHANNEL_KEYS]
            calibrated[fluo] = np.einsum(
                "ab,bjw->ajw", self.compensation_matrix, calibrated[fluo]
            )

        image = np.real(calibrated / self.lifetime_phasors[:, None, :])
        return image.astype(np.float32)


def _to_complex(values: list[float]) -> np.ndarray:
    """Convert interleaved (real, imaginary) values to a complex array."""
    a = np.asarray(values, dtype=np.float64)
    return a[0::2] + 1j * a[1::2]
