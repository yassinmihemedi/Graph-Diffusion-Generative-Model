# Third-Party Code and License Notices

This repository (`ERGM_XOM`) is a derivative work. It began as a fork of
the official code release for **EDGE: Efficient and Degree-Guided Graph
Generation via Discrete Diffusion Modeling**
(https://github.com/tufts-ml/graph-generation-EDGE, MIT License), extended
into SCR (Structural Candidate Restriction): a candidate-set-restricted,
degree-adaptive discrete diffusion process anchored to structural
sufficient statistics of the target graph (edge count, triangle count,
degree sequence).

This top-level repository is released under the **MIT License** (see
`LICENSE`). However, several individual files were adapted from, or are
directly based on, other open-source projects under their own licenses.
Those licenses apply to the specific files noted below, in addition to
(not instead of) this repository's own MIT license on our original
contributions. If you redistribute this code, please preserve the notices
below alongside the corresponding files.

## Files adapted from other MIT-licensed projects

| File(s) in this repo | Source | License |
|---|---|---|
| `diffusion/diffusion_base.py`, `diffusion/diffusion_binomial_vanilla.py`, `diffusion/diffusion_binomial_active.py` | [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | MIT |
| `eval_utils/evaluation/graph_structure_evaluation.py` | [JiaxuanYou/graph-generation](https://github.com/JiaxuanYou/graph-generation) (GraphRNN) — the file's own header states this code is "largely taken from" this repository | MIT |
| `eval_utils/evaluation/models/gin/gin.py` | [weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns) (official GIN reference implementation) | MIT |
| PRDC metric computation in `eval_utils/evaluation/gin_evaluation.py` | [clovaai/generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc) | MIT |
| Evaluation modules generally (per upstream EDGE README) | [uoguelph-mlrg/GGM-metrics](https://github.com/uoguelph-mlrg/GGM-metrics), [hheidrich/CELL](https://github.com/hheidrich/CELL) | MIT |

## Files adapted from projects under other permissive licenses

| File(s) in this repo | Source | License |
|---|---|---|
| FID computation in `eval_utils/evaluation/gin_evaluation.py` | [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) | **Apache License 2.0** |
| MMD kernel code in `eval_utils/evaluation/gin_evaluation.py` | [djsutherland/opt-mmd](https://github.com/djsutherland/opt-mmd) | **BSD 3-Clause** |
| `diffusion/ema.py` | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) (`moving_averages.py`), partially based on | **Apache License 2.0** |

These two licenses are more restrictive than a bare MIT notice in what they
require of redistributors (e.g. Apache-2.0's NOTICE-file provisions,
BSD-3-Clause's non-endorsement clause). If you redistribute this code
further, consult the original licenses linked above for the exact terms
that apply to these specific files.

## Dependency with no discoverable license — flagged, not resolved

The original EDGE repository's own README states its code was *"developed
based on"* https://github.com/ehoogeboom/multinomial_diffusion. As of the
last check (2026-09), that repository has **no license file and no license
declared in its GitHub metadata**. This means its actual redistribution
terms are not established by any explicit grant — under default copyright
law, no license implies no permission beyond what's needed to view public
source. We did not author this dependency and cannot resolve its licensing
status on the original author's behalf. **If any code logic traceable to
that repository remains in this codebase's diffusion process, treat its
licensing status as unresolved and verify directly with the original
author before relying on it in a context where licensing certainty
matters** (e.g. commercial use, or a venue with strict IP requirements).

## Explicitly excluded: ORCA (orbit-counting binary)

The upstream EDGE evaluation pipeline used a compiled `orca` binary
(Hočevar & Demšar's orbit-counting algorithm) to compute the `orbits_mmd`
metric. **We deliberately did not include the `orca` binary or its source
in this release** — its license terms were not verified, and redistributing
a compiled third-party binary without confirmed terms is exactly the kind
of thing this document exists to avoid. As a result:

- `orbits_mmd` will not be computable out of the box.
- To use it, obtain `orca` yourself from its original source, confirm its
  license permits your intended use, and place the binary at
  `eval_utils/evaluation/orca/orca` (matching the path
  `eval_utils/evaluation/graph_structure_evaluation.py`'s `Orbits.orca()`
  expects). All other metrics in this codebase do not depend on it.

## Datasets

Graph data files themselves (Cora, CiteSeer, PubMed, Polblogs, Bitcoin
Alpha/OTC, Elliptic, MalNetTiny, ZINC, etc.) are **not included** in this
repository. Each has its own license/terms of use from its original
publisher; see `README.md` for how to obtain each one and cite it
correctly. This also keeps the repository small and avoids redistributing
data whose license we have not individually verified (the Elliptic dataset
in particular has specific usage terms worth checking before
redistribution).
