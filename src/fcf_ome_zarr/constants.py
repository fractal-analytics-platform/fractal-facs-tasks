"""Constants shared across tasks."""

from typing import Final

# Backend used for all tables written by these tasks.
TABLE_BACKEND: Final = "parquet"

# Tables written by the converter
# One ROI per TIFF; each FOV holds a single FCS event
FOV_ROI_TABLE: Final = "FOV_ROI_table"
IMAGE_ROI_TABLE: Final = "image_ROI_table"
FCS_EVENTS_TABLE: Final = "fcs_events"

# Column linking tables to FCS events, i.e. the event position in the FCS file
EVENT_INDEX_KEY: Final = "event_index"

# OME-NGFF version of the OME-Zarrs written by these tasks
NGFF_VERSION: Final = "0.5"
