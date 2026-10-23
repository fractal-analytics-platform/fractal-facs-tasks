### Purpose
- Convert data from S8 Flow Cytometer (TIFF files) to OME-Zarr format.
- Optionally store the event data of the matching FCS file as a feature table.

### Outputs
- A single mosaic OME-Zarr file containing all converted images. Optionally in plate format
- If an FCS file is provided, an "fcs_events" feature table with the raw values of all
  events, indexed by "event_index" (matching the FOV ROI names).

### Limitations
- POC implementation, only tested on a single dataset
- FCS values are stored as float32, integer parameters need to be reinterpreted
