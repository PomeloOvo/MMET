import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataset_directory(dataset):
    directory_name = {"minor": "dataset1", "major": "dataset2"}[dataset]
    return os.path.join(PROJECT_ROOT, "datasets", directory_name)


def load_dataset_config(dataset):
    path = os.path.join(PROJECT_ROOT, "configs", f"{dataset}.yaml")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_features(input_csv, dataset, batch_size=None, max_length=None):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    frame = pd.read_csv(input_csv)
    required = {"sequence"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    if "uniprot_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["uniprot_id"]).copy()
    else:
        frame = frame.drop_duplicates(subset=["sequence"]).copy()

    frame["seq_len"] = frame["sequence"].astype(str).str.len()
    frame = frame.sort_values("seq_len").reset_index(drop=True)

    config = load_dataset_config(dataset)
    model_name = config["model_config"]["pretrain_model"]
    embedding_config = config["embedding"]
    batch_size = batch_size or embedding_config["batch_size"]
    max_length = max_length or embedding_config["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    sequences = frame["sequence"].astype(str).tolist()
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            batch = sequences[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(**inputs)
            embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy())

    if not embeddings:
        raise ValueError("The input CSV contains no sequences")

    output_dir = dataset_directory(dataset)
    os.makedirs(output_dir, exist_ok=True)
    embedding_path = os.path.join(output_dir, f"embeddings_{dataset}.npy")
    metadata_path = os.path.join(output_dir, f"metadata_{dataset}.csv")

    np.save(embedding_path, np.concatenate(embeddings, axis=0))
    frame.drop(columns=["seq_len"]).to_csv(metadata_path, index=False)
    print(f"Saved {len(frame)} embeddings to {embedding_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ESM embeddings for clustering")
    parser.add_argument("--dataset", choices=("minor", "major"), required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        input_csv=os.path.abspath(args.input_csv),
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
