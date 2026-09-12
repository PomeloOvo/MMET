import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from Models.meta_patchet import MetaPatchET
from utils import load_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_settings(config_path):
    settings = load_config(config_path)
    training_config = load_config(project_path(settings["training_config"]))
    settings["model_config"] = training_config["model_config"]
    settings["checkpoint_path"] = project_path(settings["checkpoint_path"])
    settings["case_study_input_csv"] = project_path(settings["case_study_input_csv"])
    settings["case_study_output_csv"] = project_path(settings["case_study_output_csv"])
    requested_device = settings.get("device", "auto")
    settings["device"] = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    return settings


def load_model(settings):
    if not os.path.isfile(settings["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {settings['checkpoint_path']}")

    model_config = settings["model_config"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrain_model"])
    model = MetaPatchET(model_config).to(settings["device"])
    checkpoint = torch.load(
        settings["checkpoint_path"],
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


def run(config_path):
    settings = load_settings(config_path)
    input_csv = settings["case_study_input_csv"]
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Case-study data not found: {input_csv}")

    model, tokenizer = load_model(settings)
    frame = pd.read_csv(input_csv)
    required = {"ec_number", "sequence", "topt"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = frame.dropna(subset=list(required))
    frame = frame.groupby(["ec_number", "sequence"], as_index=False).agg({"topt": "mean"})
    for column in ("pred_zeroshot", "pred_meta", "error_zeroshot", "error_meta", "improvement"):
        frame[column] = np.nan
    frame["role"] = "Ignored"

    mean = settings["label_mean"]
    std = settings["label_std"]
    device = settings["device"]
    max_length = settings["model_config"]["max_seq_len"]
    valid_clusters = 0

    for _, group in frame.groupby("ec_number"):
        if len(group) < 2:
            continue
        valid_clusters += 1
        median = group["topt"].median()
        support_index = (group["topt"] - median).abs().idxmin()
        query_indices = [index for index in group.index if index != support_index]
        frame.at[support_index, "role"] = "Support (WT)"
        frame.loc[query_indices, "role"] = "Query (Mutant)"

        support_sequence = frame.at[support_index, "sequence"]
        support_temperature = frame.at[support_index, "topt"]
        support_target = torch.tensor(
            [(support_temperature - mean) / std],
            dtype=torch.float32,
            device=device,
        )
        query_sequences = frame.loc[query_indices, "sequence"].astype(str).tolist()

        support_inputs = tokenizer(
            [str(support_sequence)],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        query_inputs = tokenizer(
            query_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            support_features = model.extract_features(**support_inputs)
            query_features = model.extract_features(**query_inputs)
            zero_support = model.predict_from_features(support_features)
            zero_query = model.predict_from_features(query_features)

        zero_support_real = zero_support.item() * std + mean
        frame.at[support_index, "pred_zeroshot"] = zero_support_real
        frame.at[support_index, "error_zeroshot"] = abs(
            zero_support_real - support_temperature
        )
        zero_query_real = zero_query.cpu().numpy() * std + mean
        frame.loc[query_indices, "pred_zeroshot"] = zero_query_real
        frame.loc[query_indices, "error_zeroshot"] = np.abs(
            zero_query_real - frame.loc[query_indices, "topt"].to_numpy()
        )

        fast_weight = model.pred_head.weight.clone().detach().requires_grad_(True)
        fast_bias = model.pred_head.bias.clone().detach().requires_grad_(True)
        fallback_lr = settings["case_study_fallback_inner_lr"]
        weight_lr = model.extracted_inner_lrs.get("pred_head_weight", fallback_lr)
        bias_lr = model.extracted_inner_lrs.get("pred_head_bias", fallback_lr)

        with torch.enable_grad():
            for _ in range(settings["case_study_adaptation_steps"]):
                hidden = F.layer_norm(support_features, [support_features.size(-1)])
                hidden = model.bottleneck_path(hidden) + model.bottleneck_proj(hidden)
                support_prediction = F.linear(hidden, fast_weight, fast_bias).squeeze(-1)
                loss = F.mse_loss(support_prediction, support_target)
                weight_gradient, bias_gradient = torch.autograd.grad(
                    loss,
                    [fast_weight, fast_bias],
                )
                fast_weight = fast_weight - weight_lr * weight_gradient
                fast_bias = fast_bias - bias_lr * bias_gradient

        with torch.no_grad():
            support_hidden = F.layer_norm(support_features, [support_features.size(-1)])
            support_hidden = (
                model.bottleneck_path(support_hidden) + model.bottleneck_proj(support_hidden)
            )
            meta_support = F.linear(support_hidden, fast_weight, fast_bias).item() * std + mean
            query_hidden = F.layer_norm(query_features, [query_features.size(-1)])
            query_hidden = model.bottleneck_path(query_hidden) + model.bottleneck_proj(query_hidden)
            meta_query = F.linear(query_hidden, fast_weight, fast_bias).squeeze(-1)
            meta_query = meta_query.cpu().numpy() * std + mean

        frame.at[support_index, "pred_meta"] = meta_support
        frame.at[support_index, "error_meta"] = abs(meta_support - support_temperature)
        frame.at[support_index, "improvement"] = (
            frame.at[support_index, "error_zeroshot"]
            - frame.at[support_index, "error_meta"]
        )
        frame.loc[query_indices, "pred_meta"] = meta_query
        frame.loc[query_indices, "error_meta"] = np.abs(
            meta_query - frame.loc[query_indices, "topt"].to_numpy()
        )
        frame.loc[query_indices, "improvement"] = (
            frame.loc[query_indices, "error_zeroshot"]
            - frame.loc[query_indices, "error_meta"]
        )

    output_csv = settings["case_study_output_csv"]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    frame.to_csv(output_csv, index=False)
    average_improvement = frame.loc[
        frame["role"] == "Query (Mutant)", "improvement"
    ].mean()
    print(
        f"Processed {valid_clusters} clusters; mean query error improvement: "
        f"{average_improvement:.2f} C"
    )
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMET within-cluster case study")
    parser.add_argument("--config", default="Configs/Fine_tuning.yaml")
    run(project_path(parser.parse_args().config))
