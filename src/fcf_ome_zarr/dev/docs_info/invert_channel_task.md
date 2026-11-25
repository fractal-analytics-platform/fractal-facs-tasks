### Purpose
- Invert one channel of the mosaic image, selected by label, and add it as a new channel "<label>_inv", so that objects darker than the background become bright.
- The channel is inverted around its background c as max(0, c - x). c is the median of the channel over the FOVs, leaving out the padding between FOVs.
- Only the FOVs of the "FOV_ROI_table" are inverted; the padding between FOVs stays 0.

### Outputs
- The input image, rewritten in place with the extra "<label>_inv" channel; labels, tables and FCS metadata are kept
