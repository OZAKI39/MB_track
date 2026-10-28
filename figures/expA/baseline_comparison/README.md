# Baseline Comparison

Final assets: baseline_animations_20261028-154358. Each topic contains one baseline and one our-method movie. Baseline panels use thresholds 100 and 130; tracking displays independent single-target runs. All 20 baseline runs are included in data/baseline_runs/. Only accepted ours_fixed_background results are copied to data/our_method/. The diagnostic background-rebuild run is excluded. The baseline has no merger, separation or exit classifier; the event movie shows its actual behavior in the same scenes.

## Presentation assets

- [01_microbubble_detection/01_detection_baseline.mp4](01_microbubble_detection/01_detection_baseline.mp4)
- [01_microbubble_detection/01_detection_ours.mp4](01_microbubble_detection/01_detection_ours.mp4)
- [02_tracking/02_tracking_baseline.mp4](02_tracking/02_tracking_baseline.mp4)
- [02_tracking/02_tracking_ours.mp4](02_tracking/02_tracking_ours.mp4)
- [03_merger_separation_disappearance/03_events_baseline.mp4](03_merger_separation_disappearance/03_events_baseline.mp4)
- [03_merger_separation_disappearance/03_events_ours.mp4](03_merger_separation_disappearance/03_events_ours.mp4)

## Data and provenance

Associated tables and arrays are in data/. Baseline frame maps are inside each topic. Original provenance and verification records are in provenance/. Shared scientific inputs are in ../shared_data/expA/. Original paths inside copied reports still describe the original run; the collection-level inventory.csv maps every copied file to its new location and SHA-256. Images and movies were not re-encoded.
