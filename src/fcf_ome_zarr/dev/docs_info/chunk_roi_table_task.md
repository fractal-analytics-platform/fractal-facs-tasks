### Purpose
- Build an ROI table whose ROIs cover whole chunks of the image, or whole shards if the image is sharded.
- Useful to run tasks that iterate over ROI tables, but have no direct per chunk access, chunk by chunk.
- Each ROI covers an n x n grid of chunks/shards along y and x; ROIs at the image edge are clipped to the image.
- Only the highest resolution level is considered.

### Outputs
- An ROI table with one ROI per n x n grid of chunks/shards, named "chunk_ROI_table_{n}x{n}" ("shard_ROI_table_{n}x{n}" if the image is sharded) unless a table name is given
