### Purpose
- Convert data from BD S8 and A8 Flow Cytometers (TIFF files) to OME-Zarr format.
- Supports TIFFs extracted with the standalone BD image extractor or the FlowJo plugin.
- Optionally store the event data of the matching FCS file as a feature table.

### Outputs
- A single mosaic OME-Zarr file containing all converted images. Optionally in plate format
- If an FCS file is provided, an "fcs_events" feature table with the raw values of all
  events, indexed by "event_index" (matching the FOV ROI names).

### Limitations
- POC implementation, only tested on one S8 and one A8 dataset
- FCS values are stored as float32, integer parameters need to be reinterpreted
