# Slide 3 - Background Estimation and Foreground Segmentation

Final background curve: segmentation_slides_20261028-150504 (publication-style revision). Steps 2-4 and output: segmentation_slides_20261028-115531. Data include the pixel brightness series, original frame, binary mask and intermediate arrays. The curve shows the single-video percentile stage, before combining backgrounds.

## Presentation assets

- [output_binary_mask.png](output_binary_mask.png)
- [step_2_relative_darkness.png](step_2_relative_darkness.png)
- [step_3_hysteresis.png](step_3_hysteresis.png)
- [step_4_closing_cleanup.png](step_4_closing_cleanup.png)
- [output_binary_mask.pdf](output_binary_mask.pdf)
- [step_2_relative_darkness.pdf](step_2_relative_darkness.pdf)
- [step_3_hysteresis.pdf](step_3_hysteresis.pdf)
- [step_4_closing_cleanup.pdf](step_4_closing_cleanup.pdf)
- [step_1_background_estimation.png](step_1_background_estimation.png)
- [step_1_background_estimation.pdf](step_1_background_estimation.pdf)
- [step_1_background_estimation.svg](step_1_background_estimation.svg)

## Data and provenance

Associated tables and arrays are in data/. Baseline frame maps are inside each topic. Original provenance and verification records are in provenance/. Shared scientific inputs are in ../shared_data/expA/. Original paths inside copied reports still describe the original run; the collection-level inventory.csv maps every copied file to its new location and SHA-256. Images and movies were not re-encoded.
