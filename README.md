# fcf-ome-zarr

Process UZH FCF images in OME-Zarr

Proof of concept for converting S8 images to OME-Zarr & processing them with 
Fractal tasks.

Example data being processed:
<img width="1874" height="1060" alt="s8_mosaic_segmented" src="https://github.com/user-attachments/assets/80e884c1-3808-4c21-8dcb-66977432bdc3" />

Mosaic with 10'000 FACS images:
<img width="1874" height="1060" alt="S8_mosaic_zoomed_out" src="https://github.com/user-attachments/assets/d767b404-54cd-47d7-a25d-c46040ac0c81" />

Gating of image-based features in the Fractal feature explorer:
<img width="1434" height="984" alt="gating_in_fractal_explorer" src="https://github.com/user-attachments/assets/4b5aedfb-7a02-442b-bf61-adbf6069577e" />

Feature visualization in the napari-feature-visualization plugin:
<img width="2123" height="1061" alt="s8_mosaics_feature_visualisation" src="https://github.com/user-attachments/assets/39c3ec85-dba3-4427-98d7-b5951bfe8119" />


### Dev

To update the manifest, run:
```
pixi run fractal-manifest create --package fcf-ome-zarr
```
