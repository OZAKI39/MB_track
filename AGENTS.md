# Visual output preferences

- New video, figure, and review-page text must be concise English.
- Use the actual Arial font for every visible label, including numeric region IDs.
  Do not substitute OpenCV Hershey, a generic sans-serif font, or a similar font.
  If Arial is unavailable, report the missing font instead of silently replacing it.
- Keep only labels needed to interpret the image, time, state, or event. Put detailed
  provenance and limitations in the accompanying report rather than on every frame.
- The six English video exports requested on 2026-10-26 go in the project `results`
  directory for this request only. Normal future development and validation outputs
  continue under `customer_workspace`; this exception does not change their defaults.
- Preserve original videos, tracking records, annotations, and earlier renderings.
- These are visual-output preferences. Continue to explain development results to
  the user in clear Chinese as requested by the root project instructions.

# Fixed detection/tracking comparison data

- The current fixed challenge set is
  `validation/bubble_benchmark_20261026-120008`. Verify `input_lock.json`; keep its
  two video intervals, full-resolution frames and review cases unchanged when
  comparing methods. Follow the warm-up and temporal-input protocol in
  `benchmark.json`.
- Baseline contours and IDs are unreviewed proposals. A human reference exists
  only after actual review and explicit freezing of annotation revisions.
  Preserve uncertainty and real rapid changes; do not force continuity or count
  unreviewed frames as negative examples. Never fabricate reviewer confirmation.
- These recordings were already inspected and are a development challenge set,
  not an independent held-out test set. Keep them out of automatic training.
  Reference revisions must remain identifiable; do not silently replace a frozen
  reference when the method changes.

# Focused context for current development

- For online processing work, start with the current-work section of `README.md`
  and the relevant section of `microbubble_tracking/ONLINE_TRACKING.md`.
- `.ignore` limits ordinary ripgrep searches to code and documentation. Read an
  explicitly needed run, dataset, dependency or figure by path; do not scan all
  historical outputs by default. Use `rg --no-ignore` on a specific directory
  when checking data references or cleanup safety. Hidden from search does not
  mean obsolete, unused, backed up or safe to delete.
