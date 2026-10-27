# PZT records linked to the experiment videos

These are unchanged copies of the existing experiment records. SHA-256 checks
confirmed that the original recordings exactly match `../raw/expA.mp4` and
`../raw/expB.mp4`. Source paths, file hashes and timestamps are in `manifest.json`.

| Folder | Video | Experiment | Action records | Video frames |
|---|---|---|---:|---:|
| `expA/` | `../raw/expA.mp4` | Four-PZT sequence | 20 | 1320 |
| `expB/` | `../raw/expB.mp4` | PZT1 → PZT3 → PZT1 | 14 | 1320 |

Both experiments used 2.5 MHz, 0.8 Vpp when active, and 0.001 Vpp when idle.
Idle keeps the output enabled; STOP disables it. The original plans and device
readbacks are retained for verification.

## Main files in each folder

- `actions.csv`: PZT channel, state, commanded amplitude, planned time, actual
  command time, write-return time, confirmation time, and corresponding video
  frame/time ranges. `stop_repeat` records the repeated safety shutdown; these
  rows are preserved separately from the initial `stop` records.
- `video_frames.csv`: zero-based frame index, video time, and Windows receive
  and decode times for every saved frame.
- `link_report.json`: original time-association results and precision limits.
- `timeline.png`: the existing experiment timeline.
- `afg_experiment/plan.json`: action schedule, frequency/amplitude settings and
  PZT-to-device channel mapping.
- `afg_experiment/events.jsonl`: original execution events.
- `afg_link_summary.json`: readbacks, error-queue checks, shutdown verification
  and clock checks. Other copied JSON/CSV files retain the supporting camera,
  clock and recording evidence.

## Using the times

Match the analysis table's `frame` column to `video_frames.csv`'s `frame_index`.
Use `command_video_first_frame` through `command_video_last_frame` in
`actions.csv` for the video range associated with a command. The corresponding
`command_video_video_from_s` and `command_video_video_to_s` give video times.
Confirmation has its own `confirmation_video_*` range.

Frame indices start at zero. `source_seq` is a camera-receiver sequence number,
not a saved-video frame index. Planned experiment time and video time have
different origins; use the saved mapping instead of equating them directly.

These are software time associations based on reception times. AFG confirmation
is a later stable readback, about one second after a scheduled change, not a
measurement of when the physical voltage changed. Camera exposure, encoding,
network and analog-output delays remain outside the known clock/frame bounds.
Do not subtract WSL counters from Windows counters or interpret these records
as measured physical synchronization.

Original absolute paths inside copied records are preserved. Use `manifest.json`
to locate the copied files and their sources. No hardware was operated and no
new timing alignment or interpolation was performed during this copy.
