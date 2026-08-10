# Vendored third-party repos (git submodules)

These 8 directories are **git submodules** — each records only its upstream
remote URL (in `.gitmodules`) and a pinned commit (a `160000` gitlink), not the
repo contents. Cloning this repo does **not** pull their ~200 GB; run
`git submodule update --init` to fetch them at the pinned commits.

| submodule | upstream | pinned commit / branch |
|---|---|---|
| `Diffsynth` | git@github.com:valka-ai/DiffSynth.git | `3217aa6` (causal) |
| `Wilor` | git@github.com:sahithinamala05/Wilor.git | `0ccda32` (main) |
| `actionformer` | https://github.com/happyharrycn/actionformer_release.git | `61ea7eb` (main) |
| `SAM` | https://github.com/facebookresearch/sam3.git | `5dd401d` (main) |
| `lance` | https://github.com/bytedance/Lance.git | `6f06e76` (main) |
| `dwpose_repo` | https://github.com/IDEA-Research/DWPose.git | `3dca5db` (onnx) |
| `motion-data-process` | git@github.com:valka-ai/motion-data-process.git | `61104e9` (add-initial-hand-code) |
| `FACT_actseg` | https://github.com/valka-ai/FACT_actseg.git | `c774fe1` (main) |

To use:

```
git submodule update --init                 # fetch all at pinned commits
git submodule update --init Diffsynth Wilor  # or just some
```

Note: the pinned commits must exist on each upstream remote. The private ones
(Diffsynth `causal`, Wilor, motion-data-process) require the corresponding
push/access; ensure those commits are pushed upstream or the init will fail.
