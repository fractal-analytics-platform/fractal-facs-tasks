### Purpose
- Link the labels of an existing label image (e.g. from a segmentation task of another package) to their FCS events.
- Each label must lie entirely within a single FOV of the "FOV_ROI_table".

### Outputs
- A "<label_name>_to_event" feature table mapping each label to its FCS event ("event_index" column)
