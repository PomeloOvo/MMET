"""Deterministic episodic datasets and collation for MMET."""

import hashlib
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EnzymeMetaDataset(Dataset):
    """Build support/query tasks from a cluster-partitioned CSV file."""

    def __init__(
        self,
        csv_file,
        k_shot=8,
        q_query=10,
        sample_n=None,
        mode="train",
        seed=42,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.seed = seed
        self.epoch_count = 0
        self.mode = mode
        self.k_shot = k_shot
        self.q_query = q_query
        self.sample_n = sample_n
        print(f"[{mode.upper()}] Loading {csv_file} with seed {seed}")

        try:
            self.df = pd.read_csv(csv_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset not found: {csv_file}") from exc

        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        self._normalize_column_names()

        missing_columns = {"cluster_id", "sequence"} - set(self.df.columns)
        if missing_columns:
            raise KeyError(
                f"Missing required columns {sorted(missing_columns)}; "
                f"available columns: {self.df.columns.tolist()}"
            )

        self.label_col = self._find_label_column()
        self.all_valid_ids = self.df["cluster_id"].unique().tolist()
        self.cluster_pools = {}

        if self.mode == "train":
            self._build_train_pools()

        self.reshuffle_tasks()
        if self.mode == "test":
            print(f"[TEST] Loaded {len(self.all_valid_ids)} candidate tasks")

    def _normalize_column_names(self):
        aliases = {
            "cluster_id": ["cluster_label", "cluster", "label", "clusters"],
            "sequence": ["seq", "protein_seq", "sequences", "sequence_data"],
        }
        for canonical, candidates in aliases.items():
            if canonical in self.df.columns:
                continue
            for candidate in candidates:
                if candidate in self.df.columns:
                    self.df.rename(columns={candidate: canonical}, inplace=True)
                    print(f"Renamed column '{candidate}' to '{canonical}'")
                    break

        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

    def _find_label_column(self):
        for column in ("label", "topt", "temperature", "target", "opt"):
            if column in self.df.columns:
                return column
        raise ValueError(
            "No temperature column found "
            "(expected label, topt, temperature, target, or opt)"
        )

    def _build_train_pools(self):
        for cluster_id in self.all_valid_ids:
            task_df = self.df[self.df["cluster_id"] == cluster_id]
            indices = task_df.index.tolist()
            idx_min = task_df[self.label_col].idxmin()
            idx_max = task_df[self.label_col].idxmax()

            # Keep the label extrema in the support pool for every task.
            available = list(set(indices) - {idx_min, idx_max})
            cluster_hash = int(
                hashlib.md5(str(cluster_id).encode("utf-8")).hexdigest(),
                16,
            )
            rng = random.Random(self.seed + cluster_hash)
            rng.shuffle(available)

            support_ratio = self.k_shot / (self.k_shot + self.q_query)
            support_pool_size = max(self.k_shot, int(len(indices) * support_ratio))
            support_pool = [idx_min, idx_max]
            additional_support = support_pool_size - 2

            if additional_support > 0:
                support_pool.extend(available[:additional_support])
                query_pool = available[additional_support:]
            else:
                query_pool = available

            self.cluster_pools[cluster_id] = {
                "support_pool": support_pool,
                "query_pool": query_pool,
            }

    def reshuffle_tasks(self, n_samples=None):
        """Rebuild deterministic non-overlapping training tasks for one epoch."""
        self.epoch_count += 1
        self.tasks = []
        epoch_rng = random.Random(self.seed + self.epoch_count)

        if self.mode == "train":
            for cluster_id in self.all_valid_ids:
                pools = self.cluster_pools[cluster_id]
                support_pool = list(pools["support_pool"])
                query_pool = list(pools["query_pool"])
                epoch_rng.shuffle(support_pool)
                epoch_rng.shuffle(query_pool)

                while (
                    len(support_pool) >= self.k_shot
                    and len(query_pool) >= self.q_query
                ):
                    self.tasks.append(
                        {
                            "cluster_id": cluster_id,
                            "support_indices": support_pool[: self.k_shot],
                            "query_indices": query_pool[: self.q_query],
                        }
                    )
                    support_pool = support_pool[self.k_shot :]
                    query_pool = query_pool[self.q_query :]

            epoch_rng.shuffle(self.tasks)
            target_n = n_samples if n_samples is not None else self.sample_n
            if target_n is not None and target_n < len(self.tasks):
                self.tasks = self.tasks[:target_n]
            return

        target_n = n_samples if n_samples is not None else self.sample_n
        selected_ids = self.all_valid_ids
        if target_n is not None and len(selected_ids) > target_n:
            selected_ids = random.sample(selected_ids, target_n)
        self.tasks.extend({"cluster_id": cluster_id} for cluster_id in selected_ids)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_info = self.tasks[idx]
        cluster_id = task_info["cluster_id"]
        task_df = self.df[self.df["cluster_id"] == cluster_id]

        if self.mode == "train":
            support_df = self.df.loc[task_info["support_indices"]]
            query_df = self.df.loc[task_info["query_indices"]]
        elif self.mode == "val":
            support_df, query_df = self._sample_validation_task(task_df, cluster_id)
        else:
            support_df, query_df = self._sample_test_task(task_df, cluster_id)

        return {
            "task_id": cluster_id,
            "support_seqs": support_df["sequence"].tolist(),
            "query_seqs": query_df["sequence"].tolist(),
            "support_labels": support_df[self.label_col].tolist(),
            "query_labels": query_df[self.label_col].tolist(),
        }

    def _sample_validation_task(self, task_df, cluster_id):
        seed_material = f"{cluster_id}_epoch{self.epoch_count}_seed{self.seed}"
        val_seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest(), 16)
        val_seed %= 2**32 - 1
        shuffled = task_df.sample(frac=1.0, random_state=val_seed)
        support_df = shuffled.iloc[: self.k_shot]
        query_pool = shuffled.iloc[self.k_shot :]
        query_df = query_pool.sample(
            n=self.q_query,
            replace=len(query_pool) < self.q_query,
            random_state=val_seed,
        )
        return support_df, query_df

    def _sample_test_task(self, task_df, cluster_id):
        cluster_hash = int(
            hashlib.md5(str(cluster_id).encode("utf-8")).hexdigest(),
            16,
        )
        task_seed = (cluster_hash + self.seed) % (2**32 - 1)
        shuffled = task_df.sample(frac=1.0, random_state=task_seed)
        support_df = shuffled.iloc[: self.k_shot]
        query_pool = shuffled.iloc[self.k_shot :]

        if len(query_pool) >= self.q_query:
            query_df = query_pool.iloc[: self.q_query]
        else:
            query_df = query_pool.sample(
                n=self.q_query,
                replace=True,
                random_state=task_seed,
            )
        return support_df, query_df


class MetaCollate:
    """Tokenize every task in a meta-batch."""

    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", max_len=1024):
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __call__(self, batch):
        batch = [task for task in batch if task is not None]
        if not batch:
            return None

        processed_batch = []
        for task in batch:
            support = self.tokenizer(
                task["support_seqs"],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )
            query = self.tokenizer(
                task["query_seqs"],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )
            processed_batch.append(
                {
                    "task_id": task["task_id"],
                    "sup_input_ids": support["input_ids"],
                    "sup_attention_mask": support["attention_mask"],
                    "sup_labels": torch.tensor(
                        task["support_labels"], dtype=torch.float32
                    ),
                    "qry_input_ids": query["input_ids"],
                    "qry_attention_mask": query["attention_mask"],
                    "qry_labels": torch.tensor(
                        task["query_labels"], dtype=torch.float32
                    ),
                }
            )

        return processed_batch
