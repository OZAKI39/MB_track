# GitHub synchronization

Target repository: https://github.com/OZAKI39/MB_track, branch `main`.

The authoritative working directory remains
`/home/lzy/projects/new_microbubble_experiment/customer_workspace/microbubble_tracking`.
This directory becomes the root of `MB_track`. Use the existing parent Git
repository; do not create a nested repository, source mirror, clone or worktree
for synchronization. The parent's `origin` remains `OZAKI39/MB_experiment`.

During testing, include all files in `results/`, including review annotations,
tables, arrays, reports and videos. Also include the existing input videos in
`data/raw/` and the comparison video in `history/`. Ignore only local environments
and reproducible tool caches. Do not regenerate or overwrite experiment results
as part of synchronization.

Push only when requested by the user. From the parent repository, inspect changes
and stage only the reviewed paths within this subdirectory:

```bash
cd /home/lzy/projects/new_microbubble_experiment
git status --short
git diff -- customer_workspace/microbubble_tracking
git add -- customer_workspace/microbubble_tracking/<reviewed-path>
git diff --cached --stat
git diff --cached
git commit -m "Describe the reviewed tracking changes"
git subtree split --prefix=customer_workspace/microbubble_tracking --branch mb-track-main
git push mb-track mb-track-main:main
```

The `mb-track` remote points to `https://github.com/OZAKI39/MB_track.git`.
Before pushing, verify the remote branch and the complete split file list,
including every result file. If the remote contains new independent changes,
inspect and integrate them first; never force-push over them. Preserve unrelated
local changes and never stage the entire parent workspace indiscriminately.
