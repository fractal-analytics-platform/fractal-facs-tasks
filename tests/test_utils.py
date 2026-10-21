import pytest

from fcf_ome_zarr.utils import sanitize_string


@pytest.mark.parametrize(
    "value, expected",
    [
        # Unchanged: letters, digits, underscores and dashes
        ("example_data", "example_data"),
        ("my-mosaic_v2", "my-mosaic_v2"),
        ("FSC", "FSC"),
        # Whitespace
        ("Demo recording", "Demo_recording"),
        ("Demo  recording\t2", "Demo_recording_2"),
        (" Demo recording ", "Demo_recording"),
        # Hash
        ("Plate#1", "Plate_1"),
        ("#Plate 1", "Plate_1"),
        # Parentheses
        ("Plate(1)", "Plate_1"),
        ("Plate (copy)", "Plate_copy"),
        ("LightLoss (Imaging)", "LightLoss_Imaging"),
        # Slashes
        ("Imaging_BP/534/46/LP/505", "Imaging_BP_534_46_LP_505"),
        # Mixed special characters
        ("Plate #1 (2024-05-01)", "Plate_1_2024-05-01"),
        ("S8.run", "S8_run"),
        # Only special characters
        ("###", ""),
    ],
)
def test_sanitize_string(value: str, expected: str):
    assert sanitize_string(value) == expected
