# World–Action Model Architecture

- `01_world_model_overview`: one-step network, ten-step rollout and five-member ensemble.
- `02_world_model_network_detail`: feature dimensions, message-passing block and decoder.
- PNG: 20 × 11.25 inches at the requested DPI (6000 × 3375 at the default 300 dpi).
  PDF: vector with embedded Arial. SVG: vector with Arial glyphs converted to paths,
  preserving appearance on other machines.
- `architecture_manifest.json`: source/font hashes, architecture, optional checkpoint checks,
  output hashes and text-layout checks. These figures describe implementation, not performance.

Run from the microbubble_tracking project directory:

```bash
../../.venv/bin/python -m world_model.architecture --output-dir figures/world_model_architecture_new
```

To also verify every saved weight tensor, use an existing PyTorch environment:

```bash
../.venv_baseline312/bin/python -m world_model.architecture --verify-checkpoint   --output-dir figures/world_model_architecture_verified
```

The output directory must not already exist. `--font-dir` accepts a directory containing
`arial.ttf` and `arialbd.ttf`. Missing Arial causes an error, never a substitute font.

## Reading the architecture

One region is one node. Edges connect region centres less than 150 pixels apart in both
directions; self-edges are excluded. Edge gap uses radii estimated from area, not exact
contour-to-contour distance. Node input is 21 base features plus 24 Fourier position features.
The six shape features are log(area), aspect ratio, circularity, solidity, sin(2 angle)
and cos(2 angle). Current and previous actions are step-averaged amplitudes, scaled by
0.8 Vpp; inactive outputs and amplitudes below the current 0.1 Vpp activity threshold
are encoded as zero. Frequency is not an input.

Each encoder/update/decoder MLP has three linear layers and two SiLU activations.
The processor has three blocks with distinct parameters. Each edge update includes the
old edge, sender node, receiver node and encoded global input. Updated incoming edges
are summed by receiving node. Both edge and node updates retain residual connections.
The global encoding is broadcast unchanged within a step and concatenated again at
the node decoder. There is no learned global-state update inside the processor.

The exact decoder channel order is dx, dy, log-variance-x, log-variance-y,
log-area increment, log-area log-variance, aspect increment, aspect log-variance.
The detail figure groups these channels by meaning rather than their tensor order.
Position increments are multiplied by 5; log-area/aspect increments by 0.1.
Within each member, per-step variances are accumulated during the rollout.

The same full network is reused for all ten time steps within a member. Each step is
six source frames, approximately 0.1 seconds for the current 60 fps recordings.
Positions, log-area, aspect, displacement history, wall lookup, global motion and
neighbour edges are updated from predictions. Circularity, solidity, orientation and
the image-border flag remain at their initial values. No future measured positions
are fed back during prediction, and no merge/split/birth/death module is present.
Wall features are inputs, not a hard wall constraint. Node count stays fixed within
one rollout. The diagrams contain no invented trajectories or experimental images.

Each of the five separately trained members completes its own rollout. The code
evaluates members in a Python loop; the side-by-side branches indicate independent
predictions, not guaranteed parallel hardware execution. Position/log-area/aspect
means are averaged across members. Returned position variance combines mean accumulated
within-member variance (averaged over x/y) and between-member position variance
(also averaged over x/y). Although shape uncertainty is trained internally, the public
predict() result does not currently return a combined shape-uncertainty array.

Training is outside the inference diagram: the complete ten-step prediction is scored
against eligible automatic tracks; recent motion is hidden for approximately half of
nodes during training. The no-neighbour and reviewed-training variants, and the
persistence/constant-velocity/response-map/gradient-boosting baselines, are separate
comparisons rather than stages of this core network.

## Visual references

The restrained colours, explicit dataflow and overview/detail panels are inspired by
encoder–processor–decoder presentation conventions, not copied figure artwork:

- [GNS, ICML 2020, Figure 2](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)
- [GraphCast, architecture overview](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/Learning_skillful_medium-range_global_weather_forecasting.pdf)

The authoritative architecture is the local `world_model/models.py`; this is not a
claim that the local model reproduces either reference paper's architecture or results.
