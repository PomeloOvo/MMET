# MetaPatchET

MetaPatchET is a deterministic meta-learning pipeline for protein optimum-temperature prediction. A single training entry point supports the minor and major datasets through separate YAML configurations.

## Repository layout

```text
configs/              Experiment-specific settings
models/               Model, episodic dataset, and loss definitions
weights/              Initialization and trained checkpoints
meta_train_opt.py     Training, validation, checkpointing, and fixed-seed testing
utils.py              YAML configuration helpers
```

`scatter.py` is retained locally as the reference used to verify the integrated test metrics, but it is intentionally excluded from version control.

## Data

The default configs expect these files relative to the repository root:

```text
data/divide_clusters/opt/reclustered_data.csv
data/divide_clusters/opt/clustered_data1.csv
```

Each CSV must contain a sequence column, a cluster identifier, and a temperature target. In addition to the canonical names, the loader accepts common sequence, cluster, and target aliases.

## Weights

Place the checkpoints described in `weights/README.md` in the `weights/` directory. Checkpoints use Git LFS attributes because they may be too large for ordinary Git storage. Run `git lfs install` before adding them to the repository.

## Run

Create an environment with the packages in `requirements.txt`, then run one experiment from the repository root:

```bash
python meta_train_opt.py --config configs/minor.yaml
python meta_train_opt.py --config configs/major.yaml
```

Each run creates deterministic train/validation/test splits, trains the model, saves the best validation checkpoint, and evaluates that checkpoint with the configured test seed. The minor and major test seeds are `40731` and `7775`, respectively.

Strict PyTorch determinism, explicit data-loader generators, and fixed worker seeds are enabled. Exact numerical reproduction still requires a compatible software stack, CUDA stack, and GPU architecture; export the tested server environment as a pinned lock file before the final release.
