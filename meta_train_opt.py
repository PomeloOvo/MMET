"""Unified deterministic training and evaluation entry point for MMET."""

import argparse
import os
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.stats as stats
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Models.meta_patchet import MetaPatchET
from Models.meta_opt_dataloader import EnzymeMetaDataset, MetaCollate
from Models.loss_func import DensityWeightedMSE
from utils import load_config, save_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GLOBAL_ADAPTIVE_LOSS = None


def set_global_seed(seed, strict_determinism=True):
    """Seed all RNGs and optionally reject nondeterministic PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(strict_determinism)

    mode = "strict deterministic mode" if strict_determinism else "seeded mode"
    print(f"Random seed set to {seed} ({mode})")


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_project_path(path):
    """Resolve config paths relative to the repository root, not the shell cwd."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def prepare_config(config_path):
    """Load, validate and normalize one major/minor experiment config."""
    config = load_config(config_path)
    required = [
        'model_name', 'csv_path', 'output_dir', 'weights_dir',
        'best_model_name', 'train_seed', 'split_seed', 'test_seed',
        'k_shot', 'q_query', 'inner_lr', 'meta_lr', 'adaptation_steps',
        'epochs', 'meta_batch_size', 'model_config', 'initialization'
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config fields: {', '.join(missing)}")

    required_model_fields = [
        'pretrain_model', 'context_window', 'max_seq_len', 'patch_len',
        'target_window', 'n_patch_inter_heads', 'patch_inter_kernel',
    ]
    missing_model_fields = [
        key for key in required_model_fields if key not in config['model_config']
    ]
    if missing_model_fields:
        raise KeyError(
            "Missing required model_config fields: "
            f"{', '.join(missing_model_fields)}"
        )

    config['csv_path'] = resolve_project_path(config['csv_path'])
    config['output_dir'] = resolve_project_path(config['output_dir'])
    config['weights_dir'] = resolve_project_path(config['weights_dir'])
    init_path = config['initialization'].get('checkpoint')
    if init_path:
        config['initialization']['checkpoint'] = resolve_project_path(init_path)

    requested_device = config.get('device', 'auto')
    config['device'] = (
        'cuda' if requested_device == 'auto' and torch.cuda.is_available()
        else 'cpu' if requested_device == 'auto'
        else requested_device
    )
    return config


def meta_loss_fn(
    model,
    config,
    params,
    buffers,
    sup_ids,
    sup_mask,
    sup_y_norm,
    qry_ids,
    qry_mask,
    qry_y_norm,
    create_graph=True,
):
    """Adapt task-specific parameters and return normalized query loss."""
    def fcall(curr_p, curr_b, ids, mask):
        if ids.dim() == 1:
            ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
        return functional_call(
            model,
            (curr_p, curr_b),
            args=(),
            kwargs={'input_ids': ids, 'attention_mask': mask},
        )

    curr_params = params
    curr_buffers = buffers
    for _ in range(config['adaptation_steps']):
        outputs = fcall(curr_params, curr_buffers, sup_ids, sup_mask)

        if isinstance(outputs, dict):
            pred = outputs['pred']
        elif isinstance(outputs, torch.Tensor):
            pred = outputs
        else:
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.pred

        pred = pred.squeeze()
        target = sup_y_norm.squeeze()

        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        global GLOBAL_ADAPTIVE_LOSS
        if GLOBAL_ADAPTIVE_LOSS is not None:
            loss = GLOBAL_ADAPTIVE_LOSS(pred, target)
        else:
            loss = F.mse_loss(pred, target)
        params_to_update = {
            name: param
            for name, param in curr_params.items()
            if 'inner_lrs' not in name
        }
        grads = torch.autograd.grad(
            loss,
            list(params_to_update.values()),
            create_graph=create_graph,
            allow_unused=True,
            retain_graph=True,
        )

        new_params = {}

        for name, param in curr_params.items():
            if 'inner_lrs' in name:
                new_params[name] = param
        for (name, param), grad in zip(params_to_update.items(), grads):
            if grad is not None:
                if not create_graph:
                    grad = grad.detach()
                grad = torch.clamp(grad, min=-1.0, max=1.0)
                safe_name = name.replace('.', '_')
                lr_name = f'inner_lrs.{safe_name}'
                if lr_name in curr_params:
                    adaptive_lr = curr_params[lr_name]
                else:
                    adaptive_lr = config['inner_lr']

                new_params[name] = param - adaptive_lr * grad

            else:
                new_params[name] = param

        curr_params = new_params
    outputs_q = fcall(curr_params, curr_buffers, qry_ids, qry_mask)

    if isinstance(outputs_q, dict):
        pred_q = outputs_q['pred']
    elif isinstance(outputs_q, torch.Tensor):
        pred_q = outputs_q
    else:
        pred_q = outputs_q[0] if isinstance(outputs_q, (tuple, list)) else outputs_q.pred

    pred_q = pred_q.squeeze()
    target_q = qry_y_norm.squeeze()

    if pred_q.dim() == 0:
        pred_q = pred_q.unsqueeze(0)
    if target_q.dim() == 0:
        target_q = target_q.unsqueeze(0)

    if GLOBAL_ADAPTIVE_LOSS is not None:
        loss_q = GLOBAL_ADAPTIVE_LOSS(pred_q, target_q)
    else:
        loss_q = F.mse_loss(pred_q, target_q)
    return loss_q, pred_q


def train_one_epoch(model, dataloader, meta_optimizer, config, current_epoch, scaler):
    """Train one meta-learning epoch and return task-averaged metrics."""
    model.train()
    device = config['device']
    device_type = torch.device(device).type
    accumulation_steps = config.get('gradient_accumulation_steps', 4)
    meta_optimizer.zero_grad()

    is_second_order = config.get('second_order', False)
    mode_str = "SO" if is_second_order else "FO"
    pbar = tqdm(dataloader, desc=f"Ep{current_epoch}[{mode_str}]", ncols=110, mininterval=5)

    metrics = {'loss': 0.0, 'mae': 0.0, 'acc': 0.0, 'rmse': 0.0, 'count': 0}
    label_mean = torch.tensor(config['norm_mean'], device=device)
    label_std = torch.clamp(torch.tensor(config['norm_std'], device=device), min=1e-5)
    torch.cuda.empty_cache()

    for step, batch_list in enumerate(pbar):
        if not batch_list:
            continue

        for task in batch_list:
            sup_ids = task['sup_input_ids'].to(device)
            sup_mask = task['sup_attention_mask'].to(device)
            sup_y_raw = task['sup_labels'].to(device)

            qry_ids = task['qry_input_ids'].to(device)
            qry_mask = task['qry_attention_mask'].to(device)
            qry_y_raw = task['qry_labels'].to(device)

            if sup_ids.dim() == 1:
                sup_ids = sup_ids.unsqueeze(0)
                sup_mask = sup_mask.unsqueeze(0)
                sup_y_raw = sup_y_raw.unsqueeze(0)
                qry_ids = qry_ids.unsqueeze(0)
                qry_mask = qry_mask.unsqueeze(0)
                qry_y_raw = qry_y_raw.unsqueeze(0)

            sup_y_norm = (sup_y_raw - label_mean) / label_std
            qry_y_norm = (qry_y_raw - label_mean) / label_std

            curr_params = {
                name: param
                for name, param in model.named_parameters()
                if param.requires_grad and ('pred_head' in name or 'inner_lrs' in name)
            }
            curr_buffers = dict(model.named_buffers())

            with torch.autocast(device_type=device_type, enabled=device_type == 'cuda'):
                loss, qry_preds_norm = meta_loss_fn(
                    model, config, curr_params, curr_buffers,
                    sup_ids, sup_mask, sup_y_norm,
                    qry_ids, qry_mask, qry_y_norm,
                    create_graph=is_second_order,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss_val = loss.item()
            loss = loss / (accumulation_steps * len(batch_list))
            scaler.scale(loss).backward()

            with torch.no_grad():
                if qry_preds_norm.dim() > qry_y_raw.dim():
                    qry_preds_norm = qry_preds_norm.squeeze(-1)
                pred_real = qry_preds_norm * label_std + label_mean

                abs_err = torch.abs(pred_real.float() - qry_y_raw.float())
                metrics['loss'] += loss_val
                metrics['mae'] += abs_err.mean().item()
                metrics['rmse'] += torch.sqrt(torch.mean(abs_err**2)).item()
                metrics['acc'] += (abs_err < 5.0).float().mean().item() * 100
                metrics['count'] += 1

                std_label, std_pred = qry_y_raw.std() + 1e-6, pred_real.std()
                activity_ratio = std_pred / std_label
                pbar.set_postfix(
                    {
                        "L": f"{loss_val:.3f}",
                        "MAE": f"{abs_err.mean().item():.2f}",
                        "Act": f"{activity_ratio:.2f}",
                    }
                )

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(meta_optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.get('max_grad_norm', 0.3),
            )
            scaler.step(meta_optimizer)
            scaler.update()
            meta_optimizer.zero_grad()
    avg_metrics = {
        key: value / metrics['count']
        for key, value in metrics.items()
        if key != 'count' and metrics['count'] > 0
    }

    inner_lrs = [
        param.detach()
        for name, param in model.named_parameters()
        if 'inner_lrs' in name
    ]
    if inner_lrs:
        all_lrs = torch.cat([l.flatten() for l in inner_lrs])
        print(
            f"\nMeta-SGD rates: mean={all_lrs.mean().item():.6f}, "
            f"range=[{all_lrs.min().item():.5f}, {all_lrs.max().item():.5f}]"
        )

    return (
        avg_metrics.get('loss', 0),
        avg_metrics.get('mae', 0),
        avg_metrics.get('acc', 0),
        avg_metrics.get('rmse', 0),
    )


@torch.no_grad()
def validate(model, dataloader, config):
    """Evaluate task adaptation and return aggregate metrics and predictions."""
    model.eval()
    total_rmse, total_mae, total_acc, valid_batch_count = 0, 0, 0, 0
    all_preds_list, all_targets_list = [], []
    device = config['device']
    device_type = torch.device(device).type
    label_mean = torch.tensor(config['norm_mean'], device=device)
    label_std = torch.clamp(torch.tensor(config['norm_std'], device=device), min=1e-5)

    for batch_list in tqdm(dataloader, desc="Validating"):
        if not batch_list:
            continue

        for task in batch_list:
            sup_inputs = {
                'input_ids': task['sup_input_ids'].to(device),
                'attention_mask': task['sup_attention_mask'].to(device)
            }
            sup_labels = task['sup_labels'].to(device)

            qry_inputs = {
                'input_ids': task['qry_input_ids'].to(device),
                'attention_mask': task['qry_attention_mask'].to(device)
            }
            qry_labels = task['qry_labels'].to(device)

            with torch.autocast(device_type=device_type, enabled=device_type == 'cuda'):
                sup_labels = (sup_labels - label_mean) / label_std
                qry_labels = (qry_labels - label_mean) / label_std

                sup_features = model.extract_features(**sup_inputs)
                qry_features = model.extract_features(**qry_inputs)

                fast_params = {
                    name: param.clone().detach().requires_grad_(False)
                    for name, param in model.named_parameters()
                    if 'pred_' in name and 'norm' not in name.lower() and 'ln' not in name.lower()
                }

                with torch.enable_grad():
                    differentiable_params = {
                        name: param.clone().detach().requires_grad_(True)
                        for name, param in fast_params.items()
                    }
                    for _ in range(config['adaptation_steps']):
                        sup_preds = model.predict_with_params(sup_features, differentiable_params)
                        loss = F.mse_loss(sup_preds.float(), sup_labels.float())
                        grads = torch.autograd.grad(
                            loss,
                            differentiable_params.values(),
                            allow_unused=True,
                        )

                        for (name, param), grad in zip(differentiable_params.items(), grads):
                            if grad is not None:
                                grad = torch.clamp(grad, -1.0, 1.0)
                                lr_name = f'inner_lrs.{name.replace(".", "_")}'

                                adaptive_lr = (
                                    model.inner_lrs[lr_name]
                                    if hasattr(model, 'inner_lrs') and lr_name in model.inner_lrs
                                    else config['inner_lr']
                                )
                                differentiable_params[name] = (
                                    param - adaptive_lr * grad
                                ).clone().detach().requires_grad_(True)

                qry_preds = model.predict_with_params(qry_features, differentiable_params)
                pred_temp = qry_preds * label_std + label_mean
                real_temp = qry_labels * label_std + label_mean

                if torch.isnan(pred_temp).any() or torch.isinf(pred_temp).any():
                    continue

                all_preds_list.append(pred_temp.detach().cpu())
                all_targets_list.append(real_temp.detach().cpu())

                total_rmse += np.sqrt(F.mse_loss(pred_temp.float(), real_temp.float()).item())
                total_acc += (
                    (torch.abs(pred_temp.float() - real_temp.float()) <= 5.0)
                    .float()
                    .mean()
                    .item()
                )
                total_mae += torch.abs(pred_temp.float() - real_temp.float()).mean().item()
                valid_batch_count += 1
    if len(all_preds_list) == 0:
        print("Warning: All batches returned NaN!")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, torch.tensor([]), torch.tensor([])

    all_preds = torch.cat(all_preds_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)

    mask = ~torch.isnan(all_preds) & ~torch.isnan(all_targets)
    clean_preds = all_preds[mask]
    clean_targets = all_targets[mask]

    if len(clean_targets) > 1:
        ss_res = torch.sum((clean_targets - clean_preds) ** 2)
        ss_tot = torch.sum((clean_targets - clean_targets.mean()) ** 2)
        global_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else torch.tensor(0.0)

        preds_np = clean_preds.cpu().float().numpy()
        targets_np = clean_targets.cpu().float().numpy()
        try:
            pearson_r, _ = stats.pearsonr(preds_np, targets_np)
            spearman_rho, _ = stats.spearmanr(preds_np, targets_np)
        except (ValueError, FloatingPointError):
            pearson_r, spearman_rho = 0.0, 0.0
    else:
        global_r2 = torch.tensor(0.0)
        pearson_r, spearman_rho = 0.0, 0.0

    print(
        f"Validation means: target={clean_targets.mean().item():.2f}, "
        f"prediction={clean_preds.mean().item():.2f}"
    )
    print(f"Correlations: Pearson={pearson_r:.4f}, Spearman={spearman_rho:.4f}")

    final_rmse = total_rmse / valid_batch_count if valid_batch_count > 0 else 0.0
    final_mae = total_mae / valid_batch_count if valid_batch_count > 0 else 0.0
    final_acc = (total_acc / valid_batch_count * 100) if valid_batch_count > 0 else 0.0

    return (
        final_rmse,
        global_r2.item(),
        final_mae,
        final_acc,
        pearson_r,
        spearman_rho,
        all_preds,
        all_targets,
    )


def evaluate_test_set(config, checkpoint_path, train_csv_path, test_csv_path):
    """Reproduce the fixed-seed test metrics used by scatter.py, without plotting."""
    global GLOBAL_ADAPTIVE_LOSS

    test_seed = config['test_seed']
    print(f"\nRunning fixed-seed test evaluation with seed {test_seed}")
    set_global_seed(test_seed, config.get('strict_determinism', True))
    # Match the standalone scatter.py evaluation, which uses plain MSE adaptation.
    GLOBAL_ADAPTIVE_LOSS = None

    test_config = config.copy()
    test_config['adaptation_steps'] = config.get('test_adaptation_steps', 5)
    train_df = pd.read_csv(train_csv_path)
    target_col = next(
        (
            column
            for column in ['label', 'topt', 'temperature', 'target', 'opt']
            if column in train_df.columns
        ),
        None,
    )
    if target_col is None:
        raise ValueError("No target column found for test normalization")

    train_df['raw_target'] = np.clip(train_df[target_col].values, 0.1, None)
    test_config['norm_mean'] = float(train_df['raw_target'].mean())
    test_config['norm_std'] = float(train_df['raw_target'].std())
    print(
        f"Test normalization: mean={test_config['norm_mean']:.4f}, "
        f"norm_std={test_config['norm_std']:.4f}"
    )
    test_model = MetaPatchET(test_config['model_config']).to(test_config['device'])
    for name, param in test_model.named_parameters():
        param.requires_grad = "pretrain_model" not in name

    test_model.build_meta_sgd_lrs(test_config['inner_lr'])

    checkpoint = torch.load(
        checkpoint_path,
        map_location=test_config['device'],
        weights_only=False
    )
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing_keys, unexpected_keys = test_model.load_state_dict(
        state_dict,
        strict=config.get('best_checkpoint_strict', False)
    )
    if missing_keys or unexpected_keys:
        print(
            f"Non-strict checkpoint load: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )
    test_model.eval()

    meta_collate_fn = MetaCollate(
        model_name=test_config['model_config']['pretrain_model'],
        max_len=test_config['model_config']['max_seq_len']
    )
    test_dataset = EnzymeMetaDataset(
        csv_file=test_csv_path,
        k_shot=test_config['k_shot'],
        q_query=test_config['q_query'],
        mode='test',
        sample_n=config.get('test_sample_n', 100),
        seed=test_seed
    )

    total_tasks = len(test_dataset)
    sample_size = min(config.get('test_sample_n', 100), total_tasks)
    local_rng = random.Random(test_seed)
    random_indices = local_rng.sample(range(total_tasks), sample_size)
    print("First sampled test indices:", random_indices[:10])

    test_generator = torch.Generator()
    test_generator.manual_seed(test_seed)
    test_loader = DataLoader(
        Subset(test_dataset, random_indices),
        batch_size=config.get('test_meta_batch_size', 2),
        shuffle=False,
        collate_fn=meta_collate_fn,
        num_workers=config.get('test_num_workers', 0),
        worker_init_fn=seed_worker,
        generator=test_generator
    )

    rmse, r2, mae, acc, pearson, spearman, _, _ = validate(
        test_model,
        test_loader,
        test_config
    )

    print("\nReproduced test results")
    print(f"R2      : {r2:.4f}")
    print(f"MAE     : {mae:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"ACC(+/-5): {acc:.2f}%")
    print(f"Pearson : {pearson:.4f}")
    print(f"Spearman: {spearman:.4f}")
    return rmse, r2, mae, acc, pearson, spearman


def main(config):
    """Run data splitting, training, validation, checkpointing, and testing."""
    global GLOBAL_ADAPTIVE_LOSS
    set_global_seed(config['train_seed'], config.get('strict_determinism', True))

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['weights_dir'], exist_ok=True)

    history = {
        'train_loss': [],
        'train_mae': [],
        'train_rmse': [],
        'val_mae': [],
        'val_rmse': [],
        'val_r2': [],
        'val_epochs': []
    }

    device = torch.device(config['device'])
    print(f"Experiment: {config['model_name']} | Device: {device}")
    print("Preparing deterministic dataset splits")
    temp_df = pd.read_csv(config['csv_path'])

    if 'is_valid_task' in temp_df.columns:
        temp_df = temp_df.drop(columns=['is_valid_task'])
        print("Removed legacy column: is_valid_task")

    c_col = None
    possible_names = ['cluster_id', 'cluster_label', 'cluster', 'label', 'clusters']
    for name in possible_names:
        if name in temp_df.columns:
            c_col = name
            break
    if c_col is None:
        raise KeyError("No cluster identifier column found")

    counts = temp_df[c_col].value_counts()
    min_samples = config['k_shot'] + config['q_query']
    valid_ids = counts[counts >= min_samples].index.tolist()
    if not valid_ids:
        raise ValueError(
            f"No cluster contains the required {min_samples} samples"
        )
    valid_ids.sort()
    split_rng = np.random.RandomState(config['split_seed'])
    split_rng.shuffle(valid_ids)

    total_len = len(valid_ids)
    train_split = int(0.8 * total_len)
    val_split = int(0.9 * total_len)

    train_ids = valid_ids[:train_split]
    val_ids = valid_ids[train_split:val_split]
    test_ids = valid_ids[val_split:]

    print(
        f"Dataset split: total={total_len}, train={len(train_ids)}, "
        f"validation={len(val_ids)}, test={len(test_ids)}"
    )

    print("Computing normalization statistics from the training split")
    train_df = temp_df[temp_df[c_col].isin(train_ids)].copy()

    target_col = None
    for name in ['label', 'topt', 'temperature', 'target', 'opt']:
        if name in train_df.columns:
            target_col = name
            break

    if target_col is None:
        raise ValueError("No target column found")

    train_df['raw_target'] = np.clip(
        train_df[target_col].values,
        a_min=0.1,
        a_max=None,
    )
    config['norm_mean'] = float(train_df['raw_target'].mean())
    config['norm_std'] = float(train_df['raw_target'].std())
    print(
        f"Training normalization: mean={config['norm_mean']:.4f}, "
        f"std={config['norm_std']:.4f}"
    )
    save_config(config, os.path.join(config['output_dir'], 'config.yaml'))

    print("Exporting train, validation, and test CSV files")

    split_dir = os.path.dirname(config['csv_path'])
    train_csv_path = os.path.join(split_dir, "train.csv")
    val_csv_path = os.path.join(split_dir, "val.csv")
    test_csv_path = os.path.join(split_dir, "test.csv")
    temp_df[temp_df[c_col].isin(train_ids)].to_csv(train_csv_path, index=False)
    temp_df[temp_df[c_col].isin(val_ids)].to_csv(val_csv_path, index=False)
    temp_df[temp_df[c_col].isin(test_ids)].to_csv(test_csv_path, index=False)
    train_dataset = EnzymeMetaDataset(
        train_csv_path,
        k_shot=config['k_shot'],
        q_query=config['q_query'],
        sample_n=config.get('train_sample_n', 400),
        seed=config['train_seed']
    )

    val_dataset = EnzymeMetaDataset(
        val_csv_path,
        k_shot=config['k_shot'],
        q_query=config['q_query'],
        sample_n=config.get('val_sample_n', 100),
        mode='val',
        seed=config.get('val_seed', config['train_seed'])
    )

    collate_fn = MetaCollate(
        model_name=config['model_config']['pretrain_model'],
        max_len=config['model_config']['max_seq_len']
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config['train_seed'])
    val_generator = torch.Generator()
    val_generator.manual_seed(config.get('val_seed', config['train_seed']))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['meta_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('train_num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['meta_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('eval_num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator
    )
    print("Collecting normalized training labels for density estimation")
    all_train_labels = []
    label_mean = config['norm_mean']
    label_std = max(config['norm_std'], 1e-6)
    for batch_list in train_loader:
        if not batch_list:
            continue
        for task in batch_list:
            sup_y_norm = (task['sup_labels'].numpy() - label_mean) / label_std
            qry_y_norm = (task['qry_labels'].numpy() - label_mean) / label_std
            all_train_labels.extend([sup_y_norm.flatten(), qry_y_norm.flatten()])

    if not all_train_labels:
        raise ValueError("The training split produced no meta-learning tasks")
    all_train_labels_flat = np.concatenate(all_train_labels)
    GLOBAL_ADAPTIVE_LOSS = DensityWeightedMSE(all_train_labels_flat).to(device)
    print("Density-weighted training loss initialized")

    print("Initializing model")
    model = MetaPatchET(config['model_config']).to(device)
    initialization = config['initialization']
    init_mode = initialization.get('mode', 'load').lower()
    init_weight_path = initialization.get('checkpoint')

    if init_mode == 'save':
        if not init_weight_path:
            raise ValueError("initialization.checkpoint is required in save mode")
        if os.path.exists(init_weight_path) and not initialization.get('overwrite', False):
            raise FileExistsError(
                f"Initialization checkpoint already exists: {init_weight_path}. "
                "Set initialization.overwrite=true to replace it."
            )
        print(f"Saving shared initialization checkpoint: {init_weight_path}")
        os.makedirs(os.path.dirname(init_weight_path), exist_ok=True)
        torch.save(model.state_dict(), init_weight_path)

    elif init_mode == 'load':
        if not init_weight_path:
            raise ValueError("initialization.checkpoint is required in load mode")
        print(f"Loading shared initialization checkpoint: {init_weight_path}")
        if not os.path.exists(init_weight_path):
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_weight_path}")
        init_state = torch.load(init_weight_path, map_location=device, weights_only=False)
        if isinstance(init_state, dict) and 'model_state_dict' in init_state:
            init_state = init_state['model_state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(
            init_state,
            strict=initialization.get('strict', False)
        )
        print(
            f"Initialization load complete: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )

    elif init_mode != 'seeded':
        raise ValueError("initialization.mode must be load, save, or seeded")

    print("Freezing the pretrained ESM backbone")

    for name, param in model.named_parameters():
        param.requires_grad = "pretrain_model" not in name

    model.build_meta_sgd_lrs(base_inner_lr=config['inner_lr'])
    model = model.to(device)

    base_params, fusion_params, fast_params, meta_lr_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'inner_lrs' in name:
            meta_lr_params.append(param)
        elif 'fusion_weights' in name:
            fusion_params.append(param)
        elif 'bottleneck' in name:
            fast_params.append(param)
        else:
            base_params.append(param)

    meta_lr = config['meta_lr']
    meta_optimizer = torch.optim.AdamW(
        [
            {'params': base_params},
            {'params': fusion_params, 'lr': meta_lr * 50},
            {'params': fast_params, 'lr': meta_lr * 3},
            {'params': meta_lr_params, 'lr': meta_lr, 'weight_decay': 0.0},
        ],
        lr=meta_lr,
        weight_decay=1e-5,
        betas=(0.5, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode='min', factor=0.5, patience=4
    )
    frozen_count = sum(not param.requires_grad for param in model.parameters())
    trainable_count = sum(param.requires_grad for param in model.parameters())
    print(f"Parameter tensors: trainable={trainable_count}, frozen={frozen_count}")

    mode_text = (
        "SO (Second-Order)"
        if config.get('second_order', False)
        else "FO (First-Order)"
    )
    print(f"Starting meta-training: mode={mode_text}, epochs={config['epochs']}")

    best_r2 = -float('inf')
    best_epoch = 0
    best_ckpt_path = os.path.join(config['weights_dir'], config['best_model_name'])

    patience = config.get('early_stopping_patience', 5)
    early_stop_count = 0

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    for epoch in range(config['epochs']):
        try:
            train_loss, train_mae, train_acc, train_rmse = train_one_epoch(
                model,
                train_loader,
                meta_optimizer,
                config,
                current_epoch=epoch,
                scaler=scaler,
            )
            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)
            history['train_rmse'].append(train_rmse)
            print(
                f"Epoch {epoch + 1:02d}/{config['epochs']} | "
                f"Loss: {train_loss:.4f} | MAE: {train_mae:.2f} | "
                f"RMSE: {train_rmse:.2f} | ACC: {train_acc:.2f}%"
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA out of memory; skipping epoch {epoch + 1}")
                torch.cuda.empty_cache()
                continue
            raise

        if hasattr(model, 'inter_multiview_fusion'):
            current_weights = torch.softmax(
                model.inter_multiview_fusion.fusion_weights.detach(),
                dim=0,
            )
            print(
                f"Fusion weights: mLSTM={current_weights[0].item():.2%}, "
                f"Transformer={current_weights[1].item():.2%}"
            )

        if (epoch + 1) % config.get('validation_interval', 3) == 0:
            print("Validating")
            torch.cuda.empty_cache()
            val_config = config.copy()
            val_config['adaptation_steps'] = config.get('val_adaptation_steps', 12)

            val_rmse, val_r2, val_mae, _, val_pearson, val_spearman, _, _ = validate(
                model,
                val_loader,
                val_config,
            )

            scheduler.step(val_rmse)

            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            history['val_r2'].append(val_r2)
            history['val_epochs'].append(epoch + 1)

            print(
                f"Epoch {epoch + 1:02d} | Val R2: {val_r2:.4f} | "
                f"Pearson: {val_pearson:.4f} | Spearman: {val_spearman:.4f} | "
                f"Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f}"
            )

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_epoch = epoch + 1
                early_stop_count = 0

                os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                torch.save(
                    {
                        'epoch': best_epoch,
                        'model_name': config['model_name'],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': meta_optimizer.state_dict(),
                        'val_r2': best_r2,
                        'history': history,
                        'config': config,
                    },
                    best_ckpt_path,
                )
                print(f"Saved new best model (R2={best_r2:.4f})")
            else:
                early_stop_count += 1
                print(f"No validation improvement: {early_stop_count}/{patience}")
                if early_stop_count >= patience:
                    print(
                        f"Early stopping after {patience} validation checks; "
                        f"best epoch was {best_epoch}"
                    )
                    break
        if (epoch + 1) % config.get('reshuffle_interval', 6) == 0:
            print("Reshuffling training tasks")
            train_dataset.reshuffle_tasks(n_samples=config.get('train_sample_n', 400))

    print(f"\nTraining complete: best Val R2={best_r2:.4f} at epoch {best_epoch}")

    if best_epoch == 0 or not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(
            f"No best checkpoint was produced; test evaluation cannot run: {best_ckpt_path}"
        )
    del model, meta_optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    evaluate_test_set(
        config=config,
        checkpoint_path=best_ckpt_path,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train MMET from a YAML experiment config."
    )
    parser.add_argument('--config', required=True, help='Path to Major.yaml or Minor.yaml')
    args = parser.parse_args()
    main(prepare_config(args.config))
