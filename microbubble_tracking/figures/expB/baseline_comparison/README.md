# Experiment B: baseline and our method

Six movies use the same original expB video. All labels use Arial.
The baseline shows thresholds 100 and 130, kept fixed from experiment A.
Four independently seeded targets (7, 2, 5, 23) were run at both thresholds.
White contours are the reviewed reference; blue boxes are baseline outputs.
Our movies show original automatic IDs, not manually linked IDs.

Event excerpts show merge 5 + 7 at frame 253, split 7 + 14 at frame 147,
and loss of ID 23 after its last detection at frame 312. Loss is reported
at frame 321; it does not establish bubble destruction or departure from view.
The baseline does not classify these events. Both exit events in B were
excluded from the illustration because the reviewed targets are artifacts.

Results use the frozen B background and reproduce all archived detector/tracker
rows and events. The MobileSAM channel mask was generated from B's first frame.
These development examples are not independent accuracy scores or speed benchmarks.
No hardware was used. PowerPoint playback has not been tested.
