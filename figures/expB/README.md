# Experiment B — presentation assets

Generated from the current `B_offline` results and the manual review saved on
2026-10-28 at 17:09:29. All displayed labels use English and actual Arial.
Original recordings, annotations, analysis tables and experiment A assets remain unchanged.

## Slide 3 — background and segmentation

- [Background estimation](slide_03_background_estimation_and_foreground_segmentation/background_estimation/step_1_background_estimation.png)
- [Relative darkness](slide_03_background_estimation_and_foreground_segmentation/segmentation/step_2_relative_darkness.png)
- [Hysteresis thresholding](slide_03_background_estimation_and_foreground_segmentation/segmentation/step_3_hysteresis.png)
- [Closing and cleanup](slide_03_background_estimation_and_foreground_segmentation/segmentation/step_4_closing_cleanup.png)
- [Binary mask output](slide_03_background_estimation_and_foreground_segmentation/segmentation/output_binary_mask.png)

The background curve uses actual pixel (438, 780) and the existing estimator's
330 temporal samples. It illustrates the single-video percentile stage; the saved
multi-reference background is used for segmentation. Frame 253 has no holes to fill
or small regions to remove, so those panels explicitly show an unchanged result.
Numerical arrays and the original pixel series accompany the figures. PDF versions
are included; the background curve also has an SVG version.

## Slide 4 — multi-object tracking

- [Motion prediction](slide_04_multi_object_tracking/step_1_motion_prediction.png)
- [Overlap matrix](slide_04_multi_object_tracking/step_2_overlap_matrix.png)
- [Hungarian assignment](slide_04_multi_object_tracking/step_3_hungarian_assignment.png)
- [Merge, split and distance matching](slide_04_multi_object_tracking/step_4_merge_split_distance.png)
- [Tracking output](slide_04_multi_object_tracking/output_tracking.png)

An exact replay reproduces all 17,758 archived rows and 85 recorded events.
The figures use actual intermediate masks, overlap counts and assignments.
Processing times come from the archived run, not a new end-to-end speed benchmark.

## Slide 6 — motion, shape and collective analysis

- [Full-field trajectory segmentation and PZT timeline](slide_06_motion_shape_and_collective_analysis/trajectory_segmentation/01_trajectory_segmentation.mp4)
- [Kinematics](slide_06_motion_shape_and_collective_analysis/kinematics/02_kinematics.mp4)
- [Morphology](slide_06_motion_shape_and_collective_analysis/morphology/03_morphology.mp4)
- [Collective metrics](slide_06_motion_shape_and_collective_analysis/collective_metrics/04_collective_metrics.mp4)
- [Raw-video motion validation](slide_06_motion_shape_and_collective_analysis/raw_video_validation/05_raw_video_validation.mp4)

The review contains 14,498 trusted measurements, 3,260 untrusted measurements and
zero unreviewed measurements. The analysis produces 51 trusted segments from 32
review-linked objects. Collective metrics exclude 3,014 confirmed false detections
and retain 246 uncertain measurements with their uncertainty flags.

The raw-video validation reproduces the original 100 comparisons: median vector
difference 0.549169 px, P90 2.408229 px. These compare centroid displacement with
template matching; they are not absolute tracking errors against external ground truth.
The morphology example uses frames marked suitable for shape measurement.

## Slide 8 — stimulus–response analysis

- [Observed response example](slide_08_stimulus_response_analysis/01_response_example.mp4)
- [Response summary](slide_08_stimulus_response_analysis/02_response_summary.png)
- [Per-action numbers](slide_08_stimulus_response_analysis/summary.csv)
- [Per-object decisions](slide_08_stimulus_response_analysis/responses.csv)

| Action | Responding / previously resting |
| --- | ---: |
| PZT1 on | 7 / 11 |
| PZT1 → PZT3 | 1 / 10 |
| PZT3 → PZT1 | 1 / 8 |
| All to standby | 0 / 12 |
| Output off | 0 / 12 |

No-switch descriptive control: 1 / 76 (1.3%). The numerator and denominator use
the same resting population. At the third switch, three additional responding
objects were already moving; their IDs and the original broader count remain in
the CSV rather than being included in the resting-object percentage.

## Baseline comparison

| Topic | Baseline | Our method |
| --- | --- | --- |
| Microbubble detection | [Video](baseline_comparison/videos/01_detection_baseline.mp4) | [Video](baseline_comparison/videos/01_detection_ours.mp4) |
| Tracking | [Video](baseline_comparison/videos/02_tracking_baseline.mp4) | [Video](baseline_comparison/videos/02_tracking_ours.mp4) |
| Merger, separation, disappearance | [Video](baseline_comparison/videos/03_events_baseline.mp4) | [Video](baseline_comparison/videos/03_events_ours.mp4) |

The existing MobileSAM mask generation and CSRT baseline were run on B.
Four separately initialized targets (7, 2, 5, 23) at two existing thresholds
(100, 130) give eight independent runs. The baseline has no merge/split/loss
classifier. Its seed labels do not prove that it retains the correct identity.

The disappearance example is tracking loss of ID 23, not evidence of bubble
destruction. Both recorded exit events concern reviewed artifacts and were
excluded from the examples. These are development-set illustrations, not an
independent accuracy benchmark.

## Data, checks and reproduction

- [Review and input validation](review_validation.json)
- [Final asset checks](verification.json)
- [File inventory](inventory.csv)
- [Analysis, review and PZT data snapshots](shared_data/)

Each export also contains its own provenance and per-frame mapping. The raw
source remains at `../../data/raw/expB.mp4` relative to this folder;
it has not been duplicated in this collection.

PZT frame mappings were checked against actual decoded video timestamps. They
provide software time correspondence, not calibrated exposure or acoustic onset.
MP4 files were fully decoded after encoding. PowerPoint playback itself has not
been tested. White canvases and plots are retained; video legends use translucent
backgrounds. Slow motion repeats real frames without generated intermediate images.

From the main project root, use `.venv/bin/python -m` with these modules:

- `customer_workspace.segmentation_ppt --experiment B --background-curve`
- `customer_workspace.segmentation_ppt --experiment B --frame 253`
- `customer_workspace.tracking_ppt --experiment B`
- `customer_workspace.analysis_ppt --experiment B`
- `customer_workspace.stimulus_response_ppt --experiment B`

Supply a new `--output-dir` for each export. Baseline reproduction uses
`customer_workspace.baseline_animations prepare --experiment B`, then `masks`,
`baseline`, and `render` with the same output directory. Run `masks` and `baseline`
with `customer_workspace/.venv_baseline312/bin/python`; use the project Python
for the other phases. No hardware control or scientific algorithm changes are required.
