# MMET

> Pre-release research code for enzyme optimal temperature prediction.

MMET is a sequence-only framework that combines meta-learning with patch-based, multi-view representation learning. It is designed to improve enzyme optimal temperature prediction under limited and imbalanced supervision, particularly for sparsely represented temperature ranges.

## Method Overview

MMET uses ESM-2 to encode enzyme sequences and organizes related samples into support/query meta-learning tasks. The residue embeddings are divided into fixed-length, non-overlapping patches. Bidirectional Mamba captures local dependencies within each patch, while Transformer and bidirectional mLSTM branches model complementary relationships among patches. Their fused representation is summarized by multi-head attentive pooling and mapped to the optimal temperature by a residual prediction head.

Meta-learning provides task-specific adaptation from a small labeled support set, and density-weighted regression reduces the dominance of densely represented temperature ranges during training.

## Installation

Python 3.10 and an NVIDIA GPU with a compatible CUDA environment are recommended.

```bash
git clone https://github.com/PomeloOvo/MMET.git
cd MMET
pip install -r requirements.txt
```

MMET uses the ESM-2 pretrained protein language model. If access to Hugging Face is slow or unavailable, connect to a mirror before running the scripts:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Training and Evaluation

Run the following commands from the project root. The experiment configuration is selected with `--config`. The provided configurations use `test` mode by default and load the checkpoint specified by `checkpoint_path`.

Evaluate the released checkpoints:

```bash
python meta_train_opt.py --config configs/minor.yaml
python meta_train_opt.py --config configs/major.yaml
python meta_train_opt.py --config configs/ablation.yaml
```

The mode can also be stated explicitly:

```bash
python meta_train_opt.py --config configs/minor.yaml --mode test
```

Start training by overriding the configured mode:

```bash
python meta_train_opt.py --config configs/minor.yaml --mode train
python meta_train_opt.py --config configs/major.yaml --mode train
```

Each completed training run evaluates the best validation checkpoint using the seeds in the selected configuration and reports the mean and standard deviation of the test metrics.

## Additional Experiments

Run cluster-wise fine-tuning:

```bash
python fine_tuning.py --config configs/fine_tuning.yaml
```

Run the within-cluster case study:

```bash
python case_study.py --config configs/case_study.yaml
```

Dataset paths, checkpoint paths, random seeds, model settings, training parameters, and output locations can be modified in the YAML files under `configs/`.
