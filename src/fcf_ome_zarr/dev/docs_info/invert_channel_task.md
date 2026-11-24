### Purpose
- Invert one channel of the mosaic image, selected by label, and add it as a new channel "<label>_inv".
- Unsigned integer images are inverted as dtype_max - x, float images as c - x (c defaults to 8.0, the float intensity mapped to the uint16 maximum by the converter).
- Only the FOVs of the "FOV_ROI_table" are inverted; the padding between FOVs stays 0.

### Outputs
- The input image, rewritten in place with the extra "<label>_inv" channel; labels, tables and FCS metadata are kept
