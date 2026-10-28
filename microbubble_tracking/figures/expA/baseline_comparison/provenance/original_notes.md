# Baseline and our method: result animations

Six videos use the same expA source and fixed square ROI (128, 0, 1024, 1024).
All visible text uses actual Arial. White captions/axes surround the original video.
The legend panels alone are translucent. Inputs and original results are preserved.

## Videos

- 01_detection_baseline: fixed threshold 100 (left) and 130 (right), 22 s, all 1320 source frames.
- 01_detection_ours: raw video (left), freshly recomputed segmentation (right), identical frame times.
- 02_tracking_baseline: threshold 100 / 130; independent single-target runs seeded as object 11 during 0–6 s,
  object 15 during 6–9 s, and object 10 during 9–22 s. Each tracker was actually initialized at its first trusted
  source frame and run continuously; switching the displayed run does not reinitialize it. The white outline
  is the reviewed reference; blue is the actual CSRT box/position. "Position reported" does not mean identity is correct.
- 02_tracking_ours: reviewed reference video (left), all automatic track IDs and recent 0.5-s trails (right).
  Human-linked IDs are not applied to automatic outputs. Missing-frame gaps are not interpolated.
- 03_events_baseline / 03_events_ours: the same merger (frame 201), separation (617) and exit (834) excerpts,
  played at quarter speed by repeating actual frames. The baseline scene name is context, not a baseline event.
  The baseline has no merger/split/exit classifier. Our exit label appears at frame 843, when the 8-frame
  waiting period is exceeded; its stored event frame 834 identifies the last observed position.

## Actual computation

The existing bubble_tracker.segment and Tracker were rerun over every source frame using the frozen
A_offline/background.npy. All resulting rows, IDs and events reproduce the archive exactly. The initial
attempt to rebuild the combined background changed it slightly and is retained under ours/ as a diagnostic;
it is not used in the delivered movies. The accepted outputs are under ours_fixed_background/ for this run.

The existing medany_baseline.MedanyTracker was run independently for all ten reviewed main objects at both
thresholds: 20 complete runs, not a native multi-object baseline. All per-frame outputs, including failed
tracking and reinitializations, are in baseline/. The mask was freshly generated with the existing MobileSAM
25-refinement procedure. Its prompts, model version and weight hash are in masks/provenance.json.
Threshold 130 comes from the existing project documentation; it was not retuned in this task.

Detection colors display detected pixels, not proof that every colored region is a microbubble. The default
fixed threshold, initial prompt boxes and reviewed reference all have the limitations described by the project.
These are offline result visualizations, not independent accuracy scores or controlled speed benchmarks.
No camera or PZT was operated. PowerPoint playback has not been tested in PowerPoint itself.

Reproduce from the project root using customer_workspace.baseline_animations: prepare with the project
Python, masks and baseline with customer_workspace/.venv_baseline312/bin/python, then render with the project
Python. Later phases require --output-dir pointing to the new prepare output. Dependency code is pinned to
the MobileSAM commit in masks/provenance.json; no existing project Python environment was changed.
