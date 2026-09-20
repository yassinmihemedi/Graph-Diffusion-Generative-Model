# SCR — Discrete Diffusion for Large Graph Generation via Structural Candidate Restriction

A discrete diffusion model for large-scale graph generation that restricts its forward and reverse processes to the structural sufficient statistics of the target graph, rather than treating every possible node pair as equally important.



This repository contains code and checkpoints for selected datasets. It is a derivative work built on top of [EDGE](https://github.com/tufts-ml/graph-generation-EDGE) (Chen et al., *“Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling,”* ICML 2023). See [Licensing and attribution](#licensing-and-attribution) for details.


## What's in this repository

```
├── train.py, model.py, experiment.py   # entry points
├── layers/                             # GNN backbone: GIN-style message passing,
│                                        # attention+FiLM edge decoder
├── diffusion/                          # forward/reverse diffusion process
│   ├── diffusion_base.py               # shared discrete-diffusion machinery
│   ├── diffusion_ergm*.py              # the candidate-restricted, degree-adaptive
│   │                                    # diffusion process described above (+ ER/SBM
│   │                                    # prior ablation variants)
│   ├── diffusion_binomial_*.py, diffusion_scalable.py, diffusion_partial_absorbing.py
│   ├── optim/, utils/                  # schedulers, seeding, plotting helpers
├── datasets/                           # data loading/preprocessing/evaluation glue
├── eval_utils/                         # MMD-based structural evaluation metrics,
│   └── evaluation/                     # GIN embedding evaluation, degree/cluster/orbit
│                                        # descriptors (orca/ deliberately excluded — see below)
├── requirements.txt, install_requirements.sh
├── LICENSE, THIRD_PARTY_LICENSES.md
```

## Quickstart

```bash
bash install_requirements.sh          # creates conda env `SCR` + installs everything
conda activate SCR
```

`install_requirements.sh` pins CUDA-specific wheels (PyTorch, DGL,
torch-geometric/torch-scatter) to what this codebase was developed and last
verified against — read the comments at the top of that script before
running it if your GPU needs a different CUDA version.

## Example commands

These three commands cover the full workflow. Flag values below (loss
weights, batch size, etc.) are taken from runs that produced stable
training on the named dataset — treat them as a working starting point for
a similarly-sized graph, not universal defaults.

### 1. Training on a small graph (Polblogs, N=1,222)

```bash
python train.py \
    --dataset polblogs \
    --arch GraphDenoisingERGM \
    --loss_type vb_kl \
    --parametrization x0 \
    --diffusion_dim 64 \
    --diffusion_steps 256 \
    --num_heads 8 8 8 8 1 \
    --noise_schedule cosine \
    --dp_rate 0.1 \
    --norm None \
    --batch_size 4 \
    --num_iter 64 \
    --num_workers 4 \
    --epochs 50000 \
    --eval_every 200 \
    --check_every 1000 \
    --num_generation 5 \
    --device cuda:0 \
    --lr 8e-5 \
    --optimizer adam \
    --clip_norm 1.0 \
    --disc_lambda 0.0 \
    --kl_pos_weight True \
    --rwse_k 0 \
    --sample_time_method importance \
    --daca_gamma 1.0 \
    --beta_max 1.5 \
    --lambda_LT 0.1 \
    --non_q_fraction 0.1 \
    --lambda_deg 0.5 \
    --lambda_deg_node 0.5 \
    --lambda_tri 1.0 \
    --lambda_tri_node 1.0 \
    --log_wandb False \
    --project polblogs_scr \
    --name polblogs_run1
```

### 2. Training on a large graph (PubMed, N=19,717)

Large graphs need a smaller batch (each step's candidate-pair tensor scales
with the graph itself), a lower `--beta_max` to keep enough of the graph
visible mid-schedule for the triangle signal to survive, and
`--ht_weight_max` to cap the Horvitz–Thompson reweighting from exploding on
a very sparse graph (ρ≈0.0002 here):

```bash
python -u train.py \
    --dataset pubmed \
    --arch GraphDenoisingERGM \
    --loss_type vb_kl \
    --parametrization x0 \
    --diffusion_dim 64 \
    --diffusion_steps 128 \
    --num_heads 8 8 8 8 8 1 \
    --noise_schedule cosine \
    --dp_rate 0.1 \
    --norm None \
    --batch_size 1 \
    --num_iter 16 \
    --num_workers 0 \
    --epochs 50000 \
    --eval_every 200 \
    --check_every 500 \
    --num_generation 2 \
    --device cuda:0 \
    --lr 8e-5 \
    --optimizer adam \
    --clip_norm 1.0 \
    --disc_lambda 0.0 \
    --kl_pos_weight True \
    --rwse_k 0 \
    --sample_time_method importance \
    --daca_gamma 1.0 \
    --beta_max 1.5 \
    --lambda_LT 0.01 \
    --non_q_fraction 0.002 \
    --ht_weight_max 50 \
    --lambda_deg 0.1 \
    --lambda_tri 0.05 \
    --lambda_deg_node 0.1 \
    --lambda_tri_node 0.05 \
    --log_wandb False \
    --project pubmed_scr \
    --name pubmed_run1
```

### 3. Evaluating a trained checkpoint

```bash
python eval_utils/evaluate.py \
    --run_dir ./selected_checkpoints/polblogs/ \
    --checkpoint_epoch 999 \
    --num_samples 16 \
    --batch_size 4 \
    --split test \
    --device cuda:0
```

`--run_dir` is the path `DiffusionExperiment` saved training to:
`{log_home}/{data_id}/{model_id}/{optim_id}/{run_name}`. Checkpoints are
saved as `check/checkpoint_{epoch}.pt` (0-indexed) every `--check_every`
epochs; pass `--list_checkpoints` instead of `--checkpoint_epoch` to print
what's available in a given run directory. Add `--steps_fraction 0.5` to
sample with half the trained number of reverse diffusion steps (this model
degrades gracefully under step-reduction — see the docstring in
`eval_utils/evaluate.py` for the full flag reference).

## Getting the data

No dataset files are bundled with this repository (kept out deliberately —
see [Licensing and attribution](#licensing-and-attribution)). Datasets are
loaded from `graphs/<name>.pkl` (or `graphs/<subdir>/` for the two
PyG-native datasets); create that directory and populate it as follows.

**Citation networks (`cora`, `citeseer`, `pubmed`) and `polblogs`** — each
is loaded as a single pickled `networkx.Graph` at `graphs/<name>.pkl`
(`datasets/data.py:360`). Build these from the standard sources, e.g.:

```python
import pickle, networkx as nx
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils
# cora / citeseer / pubmed
data = Planetoid(root='/tmp/planetoid', name='Cora')[0]   # or CiteSeer / PubMed
g = pyg_utils.to_networkx(data, to_undirected=True)
pickle.dump(g, open('graphs/cora.pkl', 'wb'))
```
Polblogs is available via `networkx.read_gml` on the standard
`polblogs.gml` file (Adamic & Glance, 2005) — take the largest connected
component and pickle it the same way.

**`bitcoin_alpha`, `bitcoin_otc`** — same single-pickled-graph format.
Source: the [Bitcoin OTC / Bitcoin Alpha trust-weighted signed
networks](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) on
SNAP. Build an undirected `networkx.Graph` from the edge list (drop
weights/signs unless your use case needs them — this codebase treats these
as plain structural graphs) and pickle it to `graphs/bitcoin_alpha.pkl` /
`graphs/bitcoin_otc.pkl`.

**`community`, `ego`** — a pickled **list** of `networkx.Graph` objects at
`graphs/Community.pkl` / `graphs/Ego.pkl` (`datasets/data.py:380`).
Standard synthetic community graphs / Citeseer ego-network extraction, as
used in the original GraphRNN and EDGE evaluation suites.


## Licensing and attribution

This repository is released under the **MIT License** (see `LICENSE`).
It is a derivative work incorporating code adapted from several other
open-source projects, each under its own license (mostly MIT, plus
Apache-2.0 and BSD-3-Clause in a few files) — see
**[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** for the full,
file-by-file accounting, including one upstream dependency
(`ehoogeboom/multinomial_diffusion`) whose own license status we could not
verify and are flagging honestly rather than silently assuming permissive
terms.


