# Vendored third-party repos

These 8 directories are **symlinks**, not checked-in content. Each points to a
local clone of a third-party / large repo on the research box under
`/home/ubuntu/us-west-3-fs/sahithi/_entire_code/`. Git stores only the symlink
(the target path), so cloning this repo does **not** pull ~200 GB of data/models.

| symlink | upstream / purpose |
|---|---|
| `Diffsynth` | DiffSynth Wan2.2-S2V streaming inference (valka-ai/DiffSynth, `causal` branch) |
| `Wilor` | WiLoR hand pose/detection |
| `actionformer` | ActionFormer temporal action localization |
| `SAM` | Segment Anything |
| `lance` | Lance columnar dataset store |
| `dwpose_repo` | DWPose whole-body pose estimation |
| `motion-data-process` | motion data processing utilities |
| `FACT_actseg` | FACT action segmentation |

The symlink targets are absolute paths on the research box, so they resolve
**only on that machine**. On a fresh clone elsewhere they will be dangling —
re-create each clone under `_entire_code/` (or repoint the symlink) to use them.
