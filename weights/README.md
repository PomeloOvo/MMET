# Model weights

Place the reproducibility checkpoints in this directory:

- `init_weight.pth`: shared deterministic initialization used before training.
- `best_model_minor.pth`: best checkpoint selected on the minor validation split.
- `best_model_major.pth`: best checkpoint selected on the major validation split.

The training entry point resolves these paths relative to the repository root.
The repository's `.gitattributes` tracks `.pth` files through Git LFS.
