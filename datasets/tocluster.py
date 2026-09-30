import argparse
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dataset_directory(dataset):
    directory_name = {"minor": "dataset1", "major": "dataset2"}[dataset]
    return os.path.join(PROJECT_ROOT, "datasets", directory_name)


def load_dataset_config(dataset):
    path = os.path.join(PROJECT_ROOT, "configs", f"{dataset}.yaml")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)["clustering"]


def load_centroids(features, labels, n_clusters):
    centers = np.zeros((n_clusters, features.shape[1]))
    for label in np.unique(labels):
        centers[label] = features[labels == label].mean(axis=0)
    return normalize(centers, axis=1, norm="l2")


def cluster_dataset(dataset, force=False):
    settings = load_dataset_config(dataset)
    output_dir = dataset_directory(dataset)
    embedding_path = os.path.join(output_dir, f"embeddings_{dataset}.npy")
    metadata_path = os.path.join(output_dir, f"metadata_{dataset}.csv")
    checkpoint_name = (
        f"kmeans_checkpoint_labels_k{settings['n_clusters']}_"
        f"seed{settings['random_state']}.npy"
    )
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    for path in (embedding_path, metadata_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required clustering input not found: {path}")

    features = np.load(embedding_path)
    frame = pd.read_csv(metadata_path)
    if len(features) != len(frame):
        raise ValueError("Embedding rows do not match metadata rows")
    if "sequence" not in frame.columns:
        raise KeyError("Metadata must contain a sequence column")

    target_column = next(
        (name for name in ("topt", "temperature", "label", "target") if name in frame.columns),
        None,
    )
    if target_column is None:
        raise KeyError("Metadata must contain a temperature target column")

    features = normalize(features, axis=1, norm="l2")
    n_clusters = settings["n_clusters"]
    if len(features) < n_clusters:
        raise ValueError(f"K={n_clusters} exceeds the number of samples ({len(features)})")

    if os.path.isfile(checkpoint_path) and not force:
        labels = np.load(checkpoint_path)
        if len(labels) != len(frame) or labels.min() < 0 or labels.max() >= n_clusters:
            raise ValueError("The existing K-means checkpoint is incompatible; rerun with --force")
        centroids = load_centroids(features, labels, n_clusters)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=settings["batch_size"],
            random_state=settings["random_state"],
            n_init=settings["n_init"],
            max_no_improvement=settings["max_no_improvement"],
        )
        labels = kmeans.fit_predict(features)
        np.save(checkpoint_path, labels)
        centroids = normalize(kmeans.cluster_centers_, axis=1, norm="l2")

    frame["cluster_id"] = labels
    retained_indices = []
    retained_clusters = 0
    for cluster_id in tqdm(np.unique(labels), desc="Filtering clusters"):
        indices = np.flatnonzero(labels == cluster_id)
        similarities = np.dot(features[indices], centroids[cluster_id])
        keep = (
            (similarities >= settings["min_similarity"])
            & (similarities <= settings["max_similarity"])
        )
        valid_indices = indices[keep]
        if len(valid_indices) < settings["min_samples"]:
            continue
        retained_indices.extend(valid_indices.tolist())
        retained_clusters += 1

    if not retained_indices:
        raise ValueError("No clusters remain after similarity and size filtering")

    filtered = frame.iloc[retained_indices].copy()
    remapping = {
        old_id: new_id for new_id, old_id in enumerate(filtered["cluster_id"].unique())
    }
    filtered["cluster_id"] = filtered["cluster_id"].map(remapping)
    output_path = os.path.join(output_dir, "clustered_data.csv")
    filtered.to_csv(output_path, index=False)

    print(
        f"Retained {len(filtered)}/{len(frame)} samples in {retained_clusters} clusters. "
        f"Saved to {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster normalized ESM embeddings")
    parser.add_argument("--dataset", choices=("minor", "major"), required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing K-means label checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    cluster_dataset(arguments.dataset, force=arguments.force)
