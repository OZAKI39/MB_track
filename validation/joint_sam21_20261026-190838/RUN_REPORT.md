# First joint OpenCV + SAM 2.1 run

The joint mechanism is implemented and the complete four-PZT recording was
processed. This is a **candidate pending human review**, not a demonstrated
whole-video accuracy improvement or an adopted replacement for the current GUI.

- [Full review video](review/video/index.html) · [MP4](review/video/review.mp4)
- [Fixed-frame comparisons and examples](comparison/index.html)
- [Implementation and reproduction](../../JOINT_TRACKING.md)

## Completed checks

| Check | Result |
|---|---|
| Original video | 1320 frames, 22 seconds, 1280 × 1024 |
| Joint processing | All 1320 frames completed |
| Model feedback | 5268 current-frame mask updates; 616 later model-object initializations included |
| OpenCV input consistency | All 1320 label images identical to the accepted Mix A detection results |
| Current OpenCV pixels removed | 0; current geometry is not replaced with an older shape |
| Saved final labels | All match their recorded contours |
| Feedback audit | Every update matches the recorded current mask and permitted feedback decision |
| Software checks | 38 tests passed, including 11 known-answer joint tests; formatting/static checks passed |
| Original rule tracker | Replaying all 1320 frames without model hints reproduced the old records exactly |
| Memory retention | Two 64-frame runs, 244 updates each: raw masks, final masks, decisions and feedback exactly equal with full versus 32-frame retention |
| Transfer | All 2658 downloaded artifacts match their server SHA-256 values; deployed source matches local source |
| Rendered video | Complete decode of 1320 frames; all frame times preserved; 22 seconds |

Evidence: [software validation](software_validation.json),
[rule regression](rule_regression.json), [memory comparison](memory_validation.json),
[transfer check](transfer_verification.json), [frame audit](comparison/report.json),
[video verification](review/video/report.json).

## Observed examples

**OpenCV helps SAM:** at frame 6 (0.100 s), the joint run's SAM prediction for
initial object 41 was empty. Current OpenCV geometry supplied 133 pixels, retained
the ID and was fed back to SAM. At frame 7 (0.117 s), SAM predicted 103 pixels for
that ID; initial-only SAM remained empty. The [side-by-side example](comparison/feedback_case.png)
shows the actual original pixels and recorded outputs. This is an observed
recovery, not a measurement of overall mask accuracy.

**SAM helps OpenCV:** at frame 10 (0.167 s), OpenCV did not output initial object 3.
SAM proposed 104 pixels; current dark-pixel support retained 94 pixels in the
joint output, marked uncertain. This result was not fed back into the model.
See the [missed-detection example](comparison/cv_miss_case.png).
During the first 64 frames, the original rule output lacked ID 3 in 12 frames;
the joint output contained it in every frame. These counts describe the outputs,
not human-verified identities.

**A remaining identity failure:** initial ID 3 was last observed in the joint
output at frame 476 (7.933 s). It was marked missing at frame 477 and ended after
the configured absence limit at frame 489 (8.150 s). The same location later
received other IDs. At 14.167 s, initial-only SAM attached #3 to a passing large
cluster, while the joint output used separate newer IDs for the small region and
the large cluster. Since the original ID had already ended, this is **not proof
of continuous correction through the interference**.
See [event evidence](id3_limitation.json) and the
[14.167-second detail](comparison/id3_detail_000850.png).

These are assistant-inspected examples from development recordings. No user
annotation was confirmed or frozen by this work.

## Limits and interpretation

The fixed comparison interval remains frames 360–659 (6–11 s), with causal warm-up
from frame 0. Its [per-frame table](comparison/fixed_6_11s.csv) and original fixed
keyframes are included. Extra examples above and near 14 seconds are explicitly
additional development diagnostics, not substituted benchmark intervals. The
benchmark lock remains valid. It has 0 confirmed frames and 0 confirmed
relationships; no aggregate accuracy, identity-switch rate against truth, or
improvement percentage can be reported. The ABA recording was not run in this
first joint candidate.

There were 1123 uncertain model-recovered region observations across 855 frames,
and 11 uncertain short-gap fragment groups. These are proposals for review,
not 1123 proven corrected errors. Split-candidate events decreased from 221 to
63, while merge candidates changed from 216 to 211. A lower event count may
mean fewer false splits **or missed true events**; current geometry preservation
alone does not validate the event labels. Raw OpenCV contours, raw SAM masks,
joint contours and decisions are all retained for checking this distinction.

This conservative version retains every current OpenCV pixel, so it does not
remove all false-positive detections. Current position/area checks and the
0.20-second missing timeout can still break identities. Ambiguous or model-only
results remain uncertain, and their own predictions are not used as feedback.
The full acceptance criteria therefore remain **unverified**, with observed
identity failures still present.

The RTX 4090 run took 1117.02 seconds, including model loading, decoding,
comparison and record writing: about 1.18 processed frames/second. Peak allocated
GPU memory was 2.20 GiB and process peak RAM was 36.11 GiB. Video decoding for SAM
preloads the recording into CPU memory. This is an offline implementation, not
real-time performance. Browser rendering/transfer times are separate.

Original videos, Mix A, reviewed annotations, benchmark inputs and GUI defaults
remain unchanged. No AFG/camera command, model training, Git commit or push was
performed. Only this task's remote supervisor was shut down after verified
transfer; the server and other services were left running. Large masks, videos
and generated full-frame figures remain local, with source and hash records.

To judge the first candidate, inspect the full video, the two correction examples
and the fixed difficult frames, then confirm contours and relationships in the
existing review workflow. Additional timing/model changes should be compared on
the same fixed inputs rather than promoted from raw event counts.
