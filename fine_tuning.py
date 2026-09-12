import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from transformers import AutoTokenizer

from Models.meta_patchet import MetaPatchET
from utils import load_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_settings(config_path):
    settings = load_config(config_path)
    training_config = load_config(project_path(settings["training_config"]))
    for key in (
        "checkpoint_path",
        "input_csv",
        "prepared_csv",
        "predictions_csv",
        "metrics_csv",
    ):
        settings[key] = project_path(settings[key])
    settings["model_config"] = training_config["model_config"]
    requested_device = settings.get("device", "auto")
    settings["device"] = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    return settings


def prepare_meta_tuning_data(frame, seed=42, support_fraction=0.05):
    if "cluster_id" not in frame.columns:
        raise KeyError("The input CSV must contain cluster_id")

    if "topt" not in frame.columns:
        source = next(
            (name for name in ("True_Topt", "temperature", "label") if name in frame.columns),
            None,
        )
        if source is None:
            raise KeyError("The input CSV must contain a temperature target")
        frame["topt"] = frame[source]

    frame["temp_range"] = pd.cut(
        frame["topt"],
        bins=[-np.inf, 60, 75, 90, np.inf],
        labels=["<60", "60-75", "75-90", ">=90"],
        right=False,
    )
    frame["meta_label"] = "query"

    np.random.seed(seed)
    for cluster_id in frame["cluster_id"].unique():
        indices = frame.index[frame["cluster_id"] == cluster_id]
        support_count = max(1, int(np.ceil(support_fraction * len(indices))))
        support_indices = np.random.choice(indices, size=support_count, replace=False)
        frame.loc[support_indices, "meta_label"] = "support"
    return frame


def load_meta_model(settings):
    checkpoint_path = settings["checkpoint_path"]
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_config = settings["model_config"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrain_model"])
    model = MetaPatchET(model_config).to(settings["device"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location=settings["device"],
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.extracted_inner_lrs = {
        key.removeprefix("inner_lrs."): value.item()
        for key, value in state_dict.items()
        if key.startswith("inner_lrs.")
    }
    model.eval()
    return model, tokenizer


def extract_features(model, tokenizer, sequences, settings):
    if not sequences:
        return None

    batches = []
    batch_size = settings["feature_batch_size"]
    max_length = settings["model_config"]["max_seq_len"]
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            inputs = tokenizer(
                sequences[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(settings["device"])
            batches.append(model.extract_features(**inputs).cpu())
    return torch.cat(batches, dim=0).to(settings["device"])


def calculate_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return [len(y_true), np.nan, np.nan, np.nan, np.nan, np.nan]
    return [
        len(y_true),
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
        pearsonr(y_true, y_pred)[0],
        spearmanr(y_true, y_pred)[0],
    ]


def adaptation_schedule(support_labels, settings):
    support_count = len(support_labels)
    support_mean = support_labels.mean().item()
    support_max = support_labels.max().item()
    is_high = support_max >= settings["high_temperature"]

    if is_high:
        few_support = support_count <= settings["few_support_max"]
        steps = settings["high_steps_few"] if few_support else settings["high_steps_many"]
        learning_rate = settings["high_lr_few"] if few_support else settings["high_lr_many"]
        return is_high, steps, learning_rate, settings["high_gradient_clip"]

    in_middle_range = (
        settings["middle_temperature_min"]
        <= support_mean
        <= settings["middle_temperature_max"]
    )
    if in_middle_range:
        return is_high, 0, 0.0, settings["low_gradient_clip"]
    return is_high, settings["low_steps"], settings["low_lr"], settings["low_gradient_clip"]


def adapt_parameters(model, support_features, support_labels, base_params, settings):
    label_mean = settings["label_mean"]
    label_std = settings["label_std"]
    normalized_labels = (support_labels - label_mean) / label_std
    is_high, steps, learning_rate, clip_value = adaptation_schedule(
        support_labels,
        settings,
    )
    adapted = {
        name: value.clone().detach().requires_grad_(True)
        for name, value in base_params.items()
    }

    with torch.enable_grad():
        for _ in range(steps):
            predictions = model.predict_with_params(support_features, adapted)
            sample_losses = (predictions - normalized_labels) ** 2

            if is_high:
                weights = torch.where(
                    support_labels >= settings["high_temperature"],
                    torch.tensor(settings["high_sample_weight"], device=settings["device"]),
                    torch.tensor(1.0, device=settings["device"]),
                )
                prediction_loss = torch.mean(sample_losses * weights)
                weight_penalty = sum(
                    F.mse_loss(value, base_params[name])
                    for name, value in adapted.items()
                    if "weight" in name.lower()
                )
                bias_penalty = sum(
                    F.mse_loss(value, base_params[name])
                    for name, value in adapted.items()
                    if "bias" in name.lower()
                )
                loss = (
                    prediction_loss
                    + settings["weight_regularization"] * weight_penalty
                    + settings["bias_regularization"] * bias_penalty
                )
            else:
                prediction_loss = torch.mean(sample_losses)
                penalty = sum(
                    F.mse_loss(value, base_params[name])
                    for name, value in adapted.items()
                )
                loss = prediction_loss + settings["low_regularization"] * penalty

            gradients = torch.autograd.grad(loss, adapted.values(), allow_unused=True)
            adapted = {
                name: (
                    (value - learning_rate * torch.clamp(gradient, -clip_value, clip_value))
                    .clone()
                    .detach()
                    .requires_grad_(True)
                    if gradient is not None
                    else value
                )
                for (name, value), gradient in zip(adapted.items(), gradients)
            }
    return adapted


def run(config_path):
    settings = load_settings(config_path)
    if not os.path.isfile(settings["input_csv"]):
        raise FileNotFoundError(f"Fine-tuning data not found: {settings['input_csv']}")

    frame = prepare_meta_tuning_data(
        pd.read_csv(settings["input_csv"]),
        seed=settings["split_seed"],
        support_fraction=settings["support_fraction"],
    )
    os.makedirs(os.path.dirname(settings["prepared_csv"]), exist_ok=True)
    frame.to_csv(settings["prepared_csv"], index=False)

    model, tokenizer = load_meta_model(settings)
    base_params = {
        name: parameter.clone().detach().requires_grad_(False)
        for name, parameter in model.named_parameters()
        if "pred_" in name and "norm" not in name.lower() and "ln" not in name.lower()
    }

    predictions = []
    for cluster_id in tqdm(frame["cluster_id"].dropna().unique(), desc="Fine-tuning"):
        cluster = frame[frame["cluster_id"] == cluster_id]
        support = cluster[cluster["meta_label"] == "support"]
        query = cluster[cluster["meta_label"] == "query"]
        if query.empty:
            continue

        adapted = base_params
        if not support.empty:
            support_features = extract_features(
                model,
                tokenizer,
                support["sequence"].astype(str).tolist(),
                settings,
            )
            support_labels = torch.tensor(
                support["topt"].tolist(),
                dtype=torch.float32,
                device=settings["device"],
            )
            adapted = adapt_parameters(
                model,
                support_features,
                support_labels,
                base_params,
                settings,
            )

        query_features = extract_features(
            model,
            tokenizer,
            query["sequence"].astype(str).tolist(),
            settings,
        )
        with torch.no_grad():
            normalized = model.predict_with_params(query_features, adapted)
            predicted = normalized * settings["label_std"] + settings["label_mean"]
        result = query.copy()
        result["Pred_Topt"] = predicted.cpu().numpy()
        predictions.append(result)

    if not predictions:
        raise ValueError("No query samples were produced")

    query_frame = pd.concat(predictions, ignore_index=True)
    summary = [["Global_All"] + calculate_metrics(query_frame["topt"], query_frame["Pred_Topt"])]
    for range_name in ("<60", "60-75", "75-90", ">=90"):
        subset = query_frame[query_frame["temp_range"] == range_name]
        if not subset.empty:
            summary.append([range_name] + calculate_metrics(subset["topt"], subset["Pred_Topt"]))

    metrics = pd.DataFrame(
        summary,
        columns=["Range", "Count", "MAE", "RMSE", "R2", "Pearson", "Spearman"],
    )
    os.makedirs(os.path.dirname(settings["predictions_csv"]), exist_ok=True)
    query_frame.to_csv(settings["predictions_csv"], index=False)
    metrics.to_csv(settings["metrics_csv"], index=False)
    print(metrics.to_string(index=False))
    print(f"Saved predictions to {settings['predictions_csv']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-wise MMET fine-tuning")
    parser.add_argument("--config", default="Configs/Fine_tuning.yaml")
    run(project_path(parser.parse_args().config))
