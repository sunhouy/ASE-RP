import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from activity_taxonomy import (
    ACTIVITY12_CATEGORIES,
    activity12_mapping_sha256,
    fine_to_activity12,
    validate_activity12_coverage,
)
from model import (
    FuseEmbeddings,
    POIGraph,
    Time2Vec,
    TransformerModel,
    UserEmbeddings,
)
from semantic_ranking import (
    build_poi_category_mapping,
    category_marginal_projection,
    ranking_loss,
    reversal_pairs,
    semantic_residual,
)
from semantic_teacher import (
    CACHE_SCHEMA,
    SCENARIO_PREVIEW_LENGTH,
    TEACHER_MODEL,
    TeacherCache,
    canonical_keys_for_prediction_steps,
    prepare_trajectory_dataframe,
)
from spatiotemporal_fusion import continuous_spatiotemporal_fusion


NUM_CLUSTERS = 300
POI_EMBED_DIM = 128
USER_EMBED_DIM = 128
REGION_EMBED_DIM = 64
TIME_EMBED_DIM = 64
TRANSFORMER_HEADS = 2
TRANSFORMER_HIDDEN_DIM = 1024
TRANSFORMER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.35
RESIDUAL_CLIP = 3.0
CANDIDATE_COUNT = 50
REVERSAL_THRESHOLD = 0.05
PROJECTION_CLIP = 3.0
TIME_WEIGHT = 0.10
REGION_WEIGHT = 0.10
PRIOR_SMOOTHING = 0.05
CONSISTENCY_WEIGHT = 0.05

TRAINING_STAGES = ("sequence_encoder", "ase_rp")


def parse_arguments():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", default="data/NYC/NYC_train.csv")
    parser.add_argument("--data_val", default="data/NYC/NYC_val.csv")
    parser.add_argument("--data_poi_info", default="data/NYC/poi_info.csv")
    parser.add_argument("--experiment", choices=TRAINING_STAGES, default="sequence_encoder")
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_weight", type=float, default=0.01)
    parser.add_argument("--projection_strength", type=float, default=0.02)
    parser.add_argument("--teacher_cache", default="data/NYC/activity12_teacher.json")
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    semantic_stage = args.experiment != "sequence_encoder"
    if semantic_stage and not args.load_model:
        parser.error("semantic enhancement requires --load_model")
    if not semantic_stage and args.load_model:
        parser.error("sequence encoder training does not use --load_model")
    if args.batch <= 0 or args.epochs <= 0:
        parser.error("--batch and --epochs must be positive")
    if args.semantic_weight < 0:
        parser.error("--semantic_weight must be non-negative")
    if not 0.0 <= args.projection_strength <= 1.0:
        parser.error("--projection_strength must be in [0, 1]")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_rows(rows):
    return rows


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        dataframe,
        poi_id_to_index,
        user_id_to_index,
        time_id_to_index,
        poi_to_region,
    ):
        self.rows = []
        for raw_trajectory_id, trajectory in dataframe.groupby(
            "trajectory_id", sort=True
        ):
            if len(trajectory) < 2:
                continue
            users = tuple(dict.fromkeys(trajectory["user_id"].astype(str)))
            if len(users) != 1:
                raise ValueError(
                    f"trajectory {raw_trajectory_id!r} contains multiple users"
                )
            if users[0] not in user_id_to_index:
                raise ValueError(
                    f"trajectory {raw_trajectory_id!r} contains an unseen user"
                )
            poi_ids = trajectory["POI_id"].tolist()
            missing_pois = [poi_id for poi_id in poi_ids if poi_id not in poi_id_to_index]
            if missing_pois:
                raise ValueError(
                    f"trajectory {raw_trajectory_id!r} contains POIs absent from poi_info"
                )
            time_values = trajectory["time_period"].tolist()
            missing_times = [value for value in time_values if value not in time_id_to_index]
            if missing_times:
                raise ValueError(
                    f"trajectory {raw_trajectory_id!r} contains unseen time slots"
                )
            try:
                continuous_times = [float(value) for value in time_values[:-1]]
            except (TypeError, ValueError) as exc:
                raise ValueError("time_period must be numeric") from exc
            poi_indices = [poi_id_to_index[poi_id] for poi_id in poi_ids]
            labels = poi_indices[1:]
            self.rows.append({
                "trajectory_id": str(raw_trajectory_id),
                "user_index": user_id_to_index[users[0]],
                "input_pois": poi_indices[:-1],
                "input_times": continuous_times,
                "label_pois": labels,
                "label_times": [time_id_to_index[value] for value in time_values[1:]],
                "label_regions": [poi_to_region[poi] for poi in labels],
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_transformer_masks(sequence_lengths, max_length, device):
    causal = torch.triu(
        torch.ones(max_length, max_length, device=device, dtype=torch.bool),
        diagonal=1,
    )
    lengths = torch.as_tensor(sequence_lengths, device=device)
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return causal, positions >= lengths.unsqueeze(1)


def category_distribution(counter, categories, category_to_index, device):
    values = torch.zeros(len(categories), device=device)
    for category, count in counter.items():
        if category in category_to_index and float(count) > 0:
            values[category_to_index[category]] += float(count)
    mass = values.sum()
    if mass <= 0:
        return torch.full_like(values, 1.0 / len(categories))
    return values / mass


def probability_tensor(distribution, categories, device):
    if set(distribution) != set(categories):
        return None
    try:
        values = torch.tensor(
            [float(distribution[category]) for category in categories],
            device=device,
        )
    except (TypeError, ValueError):
        return None
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        return None
    mass = values.sum()
    return values / mass if mass > 0 else None


def build_support_state(train_dataframe, category_mapper, poi_id_to_index, poi_categories):
    support_counts = Counter()
    keys_by_trajectory = {}
    final_records = {}
    target_counts = Counter()
    for raw_trajectory_id, trajectory in train_dataframe.groupby(
        "trajectory_id", sort=True
    ):
        if len(trajectory) < 2:
            continue
        trajectory_id = str(raw_trajectory_id)
        prediction_keys = canonical_keys_for_prediction_steps(trajectory)[:-1]
        keys_by_trajectory[trajectory_id] = prediction_keys
        for position, key in enumerate(prediction_keys):
            support_counts[key] += 1
            target_counts[category_mapper(
                trajectory.iloc[position + 1]["POI_catname"]
            )] += 1
        target_poi = poi_id_to_index[trajectory.iloc[-1]["POI_id"]]
        final_records[trajectory_id] = {
            "key": prediction_keys[-1],
            "target_category": category_mapper(poi_categories[target_poi]),
        }
    training_gate = {
        trajectory_id: tuple(support_counts[key] - 1 == 0 for key in keys)
        for trajectory_id, keys in keys_by_trajectory.items()
    }
    return support_counts, training_gate, final_records, target_counts


def inference_gate(dataframe, support_counts):
    result = {}
    for raw_trajectory_id, trajectory in dataframe.groupby(
        "trajectory_id", sort=True
    ):
        if len(trajectory) < 2:
            continue
        keys = canonical_keys_for_prediction_steps(trajectory)[:-1]
        result[str(raw_trajectory_id)] = tuple(
            support_counts.get(key, 0) == 0 for key in keys
        )
    return result


def zero_support_mask(gate, trajectory_id, sequence_length):
    mask = tuple(gate.get(str(trajectory_id), ()))
    if len(mask) != int(sequence_length):
        raise ValueError(
            f"zero-support mask length mismatch for {trajectory_id!r}"
        )
    return mask


def metric_state():
    return {
        "count": 0,
        "acc@1": 0.0,
        "acc@5": 0.0,
        "acc@10": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
    }


def update_metrics(state, logits, labels, sequence_lengths):
    for row, length in enumerate(sequence_lengths):
        scores = logits[row, :length].detach().float()
        targets = labels[row, :length]
        target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        state["count"] += int(length)
        state["acc@1"] += float((ranks <= 1).sum())
        state["acc@5"] += float((ranks <= 5).sum())
        state["acc@10"] += float((ranks <= 10).sum())
        state["mrr"] += float((1.0 / ranks.float()).sum())
        state["ndcg@5"] += float(
            torch.where(
                ranks <= 5,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            ).sum()
        )
        state["ndcg@10"] += float(
            torch.where(
                ranks <= 10,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float),
            ).sum()
        )


def summarize_metrics(state):
    count = int(state["count"])
    denominator = max(count, 1)
    return {
        key: float(state[key]) / denominator
        for key in ("acc@1", "acc@5", "acc@10", "mrr", "ndcg@5", "ndcg@10")
    } | {"count": count}


def metric_table(title, metrics, loss=None):
    lines = [
        title,
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Acc@1 | {metrics['acc@1']:.6f} |",
        f"| Acc@5 | {metrics['acc@5']:.6f} |",
        f"| Acc@10 | {metrics['acc@10']:.6f} |",
        f"| MRR | {metrics['mrr']:.6f} |",
        f"| NDCG@5 | {metrics['ndcg@5']:.6f} |",
        f"| NDCG@10 | {metrics['ndcg@10']:.6f} |",
        f"| Count | {metrics['count']} |",
    ]
    if loss is not None:
        lines.append(f"| Loss | {loss:.6f} |")
    return "\n".join(lines)


def category_metric_table(title, metrics):
    return "\n".join([
        title,
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MRR | {metrics['mrr']:.6f} |",
        f"| Count | {metrics['count']} |",
    ])


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.semantic_stage = args.experiment != "sequence_encoder"
        self.train_df = prepare_trajectory_dataframe(
            pd.read_csv(args.data_train), require_geo=True
        )
        self.val_df = prepare_trajectory_dataframe(
            pd.read_csv(args.data_val), require_geo=True
        )
        self.poi_df = pd.read_csv(args.data_poi_info)
        self._build_data_state()
        self._build_support_state()
        self._build_datasets()
        self._build_models()
        self.reference_summary = {}
        if self.semantic_stage:
            self._load_reference_checkpoint()
            self._load_semantic_supervision()
            self._freeze_representation()
        else:
            self.semantic_payloads = {}
        self._move_models()
        self._cache_fixed_embeddings()
        if self.semantic_stage:
            self._build_candidate_pairs()
        self._build_optimizer()

    def _build_data_state(self):
        required = {"poi_id", "longitude", "latitude"}
        missing = sorted(required.difference(self.poi_df.columns))
        if missing:
            raise ValueError(f"poi_info missing columns: {missing}")
        if self.poi_df["poi_id"].duplicated().any():
            raise ValueError("poi_info contains duplicate POI identifiers")
        self.poi_ids = self.poi_df["poi_id"].tolist()
        self.poi_id_to_index = {
            poi_id: index for index, poi_id in enumerate(self.poi_ids)
        }
        self.num_pois = len(self.poi_ids)
        if self.num_pois < NUM_CLUSTERS:
            raise ValueError(
                f"300 regions require at least {NUM_CLUSTERS} POIs"
            )
        coordinates = self.poi_df[["longitude", "latitude"]].to_numpy()
        if not np.isfinite(coordinates.astype(float)).all():
            raise ValueError("poi_info coordinates must be finite")
        region_labels = KMeans(
            n_clusters=NUM_CLUSTERS,
            max_iter=1000,
            random_state=self.args.seed,
            n_init=10,
        ).fit_predict(coordinates)
        self.poi_to_region = {
            index: int(region_labels[index]) for index in range(self.num_pois)
        }
        self.num_regions = NUM_CLUSTERS
        self.user_ids = sorted(
            self.train_df["user_id"].astype(str).unique().tolist(), key=str
        )
        self.user_id_to_index = {
            user_id: index for index, user_id in enumerate(self.user_ids)
        }
        self.num_users = len(self.user_ids)
        self.time_ids = sorted(
            self.train_df["time_period"].unique().tolist(), key=str
        )
        self.time_id_to_index = {
            time_id: index for index, time_id in enumerate(self.time_ids)
        }
        self.num_times = len(self.time_ids)
        visits = pd.concat((self.train_df, self.val_df), ignore_index=True)
        self.poi_categories = build_poi_category_mapping(
            visits, self.num_pois, self.poi_id_to_index
        )
        if "poi_catid" in self.poi_df and "POI_catid" in visits:
            # Recover catalog categories for POIs absent from both splits.
            names = visits.groupby("POI_catid")["POI_catname"]
            unambiguous = names.nunique().eq(1)
            names_by_id = names.first().loc[unambiguous]
            for index, category_id in enumerate(self.poi_df["poi_catid"]):
                if index not in self.poi_categories and category_id in names_by_id:
                    self.poi_categories[index] = str(names_by_id[category_id])
        if len(self.poi_categories) != self.num_pois:
            missing_count = self.num_pois - len(self.poi_categories)
            raise ValueError(f"{missing_count} POIs have no category metadata")
        validate_activity12_coverage(set(self.poi_categories.values()))
        self.categories = list(ACTIVITY12_CATEGORIES)
        self.category_to_index = {
            category: index for index, category in enumerate(self.categories)
        }
        self.num_categories = len(self.categories)
        self.taxonomy_fingerprint = activity12_mapping_sha256()
        self.poi_category_index = torch.tensor([
            self.category_to_index[fine_to_activity12(self.poi_categories[index])]
            for index in range(self.num_pois)
        ], dtype=torch.long, device=self.device)
        self.user_poi = np.zeros(
            (self.num_users, self.num_pois), dtype=np.float32
        )
        user_time_poi = np.zeros(
            (self.num_users, self.num_times, self.num_pois), dtype=np.float32
        )
        for row in self.train_df[
            ["user_id", "POI_id", "time_period"]
        ].itertuples(index=False):
            user = self.user_id_to_index[str(row.user_id)]
            poi = self.poi_id_to_index[row.POI_id]
            slot = self.time_id_to_index[row.time_period]
            self.user_poi[user, poi] += 1.0
            user_time_poi[user, slot, poi] += 1.0
        user_mass = self.user_poi.sum(axis=1, keepdims=True)
        np.divide(
            self.user_poi,
            user_mass,
            out=self.user_poi,
            where=user_mass > 0,
        )
        time_mass = user_time_poi.sum(axis=-1, keepdims=True)
        np.divide(
            user_time_poi,
            time_mass,
            out=user_time_poi,
            where=time_mass > 0,
        )
        self.user_poi = torch.from_numpy(self.user_poi).to(self.device)
        self.user_time_poi = torch.from_numpy(user_time_poi).to(self.device)
        poi_edges = [[], []]
        region_edges = [[], []]
        for _, trajectory in self.train_df.groupby("trajectory_id", sort=True):
            indices = [
                self.poi_id_to_index[poi_id]
                for poi_id in trajectory["POI_id"].tolist()
            ]
            for source, target in zip(indices[:-1], indices[1:]):
                poi_edges[0].append(source)
                poi_edges[1].append(target)
                region_edges[0].append(self.poi_to_region[source])
                region_edges[1].append(self.poi_to_region[target])
        self.poi_edges = torch.tensor(
            poi_edges, dtype=torch.long, device=self.device
        )
        self.region_edges = torch.tensor(
            region_edges, dtype=torch.long, device=self.device
        )
        self.poi_indices = torch.arange(self.num_pois, device=self.device)
        self.user_indices = torch.arange(self.num_users, device=self.device)
        self.region_indices = torch.arange(self.num_regions, device=self.device)
        self.poi_region_index = torch.tensor([
            self.poi_to_region[index] for index in range(self.num_pois)
        ], dtype=torch.long, device=self.device)

    def _build_support_state(self):
        (
            self.support_counts,
            self.train_gate,
            self.final_records,
            self.global_category_counts,
        ) = build_support_state(
            self.train_df,
            fine_to_activity12,
            self.poi_id_to_index,
            self.poi_categories,
        )
        self.val_gate = inference_gate(self.val_df, self.support_counts)
        print(json.dumps({
            "zero_support": {
                "train": [
                    sum(sum(mask) for mask in self.train_gate.values()),
                    sum(len(mask) for mask in self.train_gate.values()),
                ],
                "validation": [
                    sum(sum(mask) for mask in self.val_gate.values()),
                    sum(len(mask) for mask in self.val_gate.values()),
                ],
            }
        }, sort_keys=True))

    def _build_datasets(self):
        arguments = (
            self.poi_id_to_index,
            self.user_id_to_index,
            self.time_id_to_index,
            self.poi_to_region,
        )
        self.train_dataset = TrajectoryDataset(self.train_df, *arguments)
        self.val_dataset = TrajectoryDataset(self.val_df, *arguments)
        use_cuda = self.device.type == "cuda"
        workers = self.args.num_workers if use_cuda else 0
        loader_options = {
            "batch_size": self.args.batch,
            "num_workers": workers,
            "pin_memory": use_cuda,
            "persistent_workers": workers > 0,
            "collate_fn": collate_rows,
        }
        self.train_loader = DataLoader(
            self.train_dataset, shuffle=True, **loader_options
        )
        self.train_reference_loader = DataLoader(
            self.train_dataset, shuffle=False, **loader_options
        )
        self.val_loader = DataLoader(
            self.val_dataset, shuffle=False, **loader_options
        )

    def _build_models(self):
        self.poi_model = POIGraph(self.num_pois, POI_EMBED_DIM)
        self.user_model = UserEmbeddings(self.num_users, USER_EMBED_DIM)
        self.time_model = Time2Vec(TIME_EMBED_DIM)
        self.region_model = POIGraph(self.num_regions, REGION_EMBED_DIM)
        self.fusion_model = FuseEmbeddings(
            POI_EMBED_DIM + USER_EMBED_DIM,
            TIME_EMBED_DIM + REGION_EMBED_DIM,
        )
        embedding_dim = (
            POI_EMBED_DIM + USER_EMBED_DIM + REGION_EMBED_DIM + TIME_EMBED_DIM
        )
        self.sequence_model = TransformerModel(
            self.num_pois,
            self.num_regions,
            self.num_times,
            embedding_dim,
            TRANSFORMER_HEADS,
            TRANSFORMER_HIDDEN_DIM,
            TRANSFORMER_LAYERS,
            self.num_categories,
            TRANSFORMER_DROPOUT,
        )
        self.modules = {
            "poi_model": self.poi_model,
            "user_model": self.user_model,
            "time_model": self.time_model,
            "region_model": self.region_model,
            "fusion_model": self.fusion_model,
            "sequence_model": self.sequence_model,
        }

    def model_config(self):
        return {
            "num_clusters": NUM_CLUSTERS,
            "poi_embed_dim": POI_EMBED_DIM,
            "user_embed_dim": USER_EMBED_DIM,
            "region_embed_dim": REGION_EMBED_DIM,
            "time_embed_dim": TIME_EMBED_DIM,
            "transformer_heads": TRANSFORMER_HEADS,
            "transformer_hidden_dim": TRANSFORMER_HIDDEN_DIM,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_dropout": TRANSFORMER_DROPOUT,
            "region_weight": REGION_WEIGHT,
            "time_weight": TIME_WEIGHT,
            "prior_smoothing": PRIOR_SMOOTHING,
            "consistency_weight": CONSISTENCY_WEIGHT,
        }

    def _load_reference_checkpoint(self):
        path = Path(self.args.load_model)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "sequence_encoder":
            raise ValueError("reference checkpoint must contain a sequence encoder")
        if checkpoint.get("model_config") != self.model_config():
            raise ValueError("reference checkpoint model configuration differs")
        expected = {
            "poi_id_to_index": self.poi_id_to_index,
            "user_id_to_index": self.user_id_to_index,
            "time_id_to_index": self.time_id_to_index,
            "poi_to_region": self.poi_to_region,
            "support_counts": dict(self.support_counts),
            "category_mapping_sha256": self.taxonomy_fingerprint,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"reference checkpoint field {key!r} differs")
        for name, module in self.modules.items():
            if name not in checkpoint:
                raise ValueError(f"reference checkpoint is missing {name}")
            module.load_state_dict(checkpoint[name], strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        validation = checkpoint.get("validation_metrics") or {}
        validation_mrr = float(validation.get("mrr", float("nan")))
        if not np.isfinite(validation_mrr):
            raise ValueError("reference checkpoint has no validation MRR")
        self.reference_summary = {
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "epoch": int(checkpoint.get("epoch", -1)),
            "validation_mrr": validation_mrr,
        }

    def _validate_teacher_cache(self, cache):
        fingerprints = set()
        errors = set()
        category_set = set(self.categories)
        include_supported = set()
        for entry in cache.entries.values():
            config = entry.get("generation_config", {})
            if entry.get("cache_schema") != CACHE_SCHEMA:
                errors.add("cache_schema")
            if config.get("teacher_model") != TEACHER_MODEL:
                errors.add("teacher_model")
            if config.get("taxonomy_name") != "activity12":
                errors.add("taxonomy_name")
            if config.get("taxonomy_mapping_sha256") != self.taxonomy_fingerprint:
                errors.add("taxonomy_mapping")
            if int(config.get("scenario_preview_length", -1)) != SCENARIO_PREVIEW_LENGTH:
                errors.add("scenario_preview_length")
            if int(config.get("category_count", -1)) != len(self.categories):
                errors.add("category_count")
            include_supported.add(bool(config.get("include_supported", False)))
            if entry.get("valid", False) and set(entry.get("distribution", {})) != category_set:
                errors.add("distribution")
            fingerprint = entry.get("generation_fingerprint")
            if fingerprint:
                fingerprints.add(fingerprint)
            else:
                errors.add("generation_fingerprint")
        if include_supported != {False}:
            errors.add("include_supported")
        if errors or len(fingerprints) != 1:
            raise ValueError(
                "invalid teacher cache: "
                + ", ".join(sorted(errors or {"generation_fingerprint"}))
            )

    def _load_semantic_supervision(self):
        cache_path = Path(self.args.teacher_cache)
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache = TeacherCache(
            cache_path=str(cache_path),
            taxonomy_fingerprint=self.taxonomy_fingerprint,
        )
        cache.load()
        self._validate_teacher_cache(cache)
        distributions = {}
        for trajectory_id, record in self.final_records.items():
            entry = cache.entries.get(trajectory_id)
            if not entry or not entry.get("valid", False):
                continue
            expected_support = self.support_counts[record["key"]] - 1
            if entry.get("pattern_key") != record["key"]:
                raise ValueError(
                    f"teacher context differs for trajectory {trajectory_id!r}"
                )
            if int(entry.get("support_without_self", -1)) != expected_support:
                raise ValueError(
                    f"teacher support differs for trajectory {trajectory_id!r}"
                )
            if entry.get("true_category") != record["target_category"]:
                raise ValueError(
                    f"teacher target metadata differs for trajectory {trajectory_id!r}"
                )
            distribution = probability_tensor(
                entry["distribution"], self.categories, self.device
            )
            if distribution is not None:
                distributions[trajectory_id] = distribution
        self.semantic_payloads = {}
        for trajectory_id, record in self.final_records.items():
            if trajectory_id not in distributions:
                continue
            if self.support_counts[record["key"]] - 1 != 0:
                continue
            empirical_counts = self.global_category_counts.copy()
            empirical_counts[record["target_category"]] -= 1
            if empirical_counts[record["target_category"]] <= 0:
                del empirical_counts[record["target_category"]]
            empirical = category_distribution(
                empirical_counts,
                self.categories,
                self.category_to_index,
                self.device,
            )
            residual = semantic_residual(
                distributions[trajectory_id], empirical, clip=RESIDUAL_CLIP
            )
            self.semantic_payloads[trajectory_id] = {"residual": residual}
        supervision_count = len(self.semantic_payloads)
        if supervision_count == 0:
            raise RuntimeError("no valid semantic supervision was found")
        print(json.dumps({
            "experiment": self.args.experiment,
            "semantic_supervision": supervision_count,
        }, sort_keys=True))

    def _freeze_representation(self):
        for module in self.modules.values():
            for parameter in module.parameters():
                parameter.requires_grad = False
        for head in (
            self.sequence_model.decoder_poi,
            self.sequence_model.decoder_region,
            self.sequence_model.decoder_time,
            self.sequence_model.decoder_cat,
        ):
            for parameter in head.parameters():
                parameter.requires_grad = True

    def _move_models(self):
        for module in self.modules.values():
            module.to(self.device)

    def _cache_fixed_embeddings(self):
        self.fixed_embeddings = None
        if not self.semantic_stage:
            return
        for module in (self.poi_model, self.region_model, self.user_model):
            module.eval()
        with torch.no_grad():
            poi = self.poi_model(self.poi_indices, self.poi_edges)
            region = self.region_model(self.region_indices, self.region_edges)
            user = self.user_model(
                self.user_indices, torch.mm(self.user_poi, poi)
            )
        self.fixed_embeddings = (poi.detach(), region.detach(), user.detach())

    def _global_embeddings(self):
        if self.fixed_embeddings is not None:
            return self.fixed_embeddings
        poi = self.poi_model(self.poi_indices, self.poi_edges)
        region = self.region_model(self.region_indices, self.region_edges)
        user = self.user_model(
            self.user_indices, torch.mm(self.user_poi, poi)
        )
        return poi, region, user

    def _forward_batch(self, rows):
        lengths = [len(row["input_pois"]) for row in rows]
        trajectory_ids = [row["trajectory_id"] for row in rows]
        users = [row["user_index"] for row in rows]
        flat_pois = [poi for row in rows for poi in row["input_pois"]]
        flat_times = [value for row in rows for value in row["input_times"]]
        flat_regions = [
            self.poi_to_region[poi]
            for row in rows
            for poi in row["input_pois"]
        ]
        flat_users = [
            row["user_index"]
            for row in rows
            for _ in row["input_pois"]
        ]
        poi_index = torch.tensor(flat_pois, device=self.device)
        region_index = torch.tensor(flat_regions, device=self.device)
        user_index = torch.tensor(flat_users, device=self.device)
        time_values = torch.tensor(
            flat_times, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        poi_embeddings, region_embeddings, user_embeddings = (
            self._global_embeddings()
        )
        fused = self.fusion_model(
            torch.cat((user_embeddings[user_index], poi_embeddings[poi_index]), dim=-1),
            torch.cat((self.time_model(time_values), region_embeddings[region_index]), dim=-1),
        )
        inputs = pad_sequence(
            list(torch.split(fused, lengths)),
            batch_first=True,
            padding_value=-1,
        )
        labels = {}
        for name, key in (
            ("poi", "label_pois"),
            ("time", "label_times"),
            ("region", "label_regions"),
        ):
            values = torch.tensor(
                [value for row in rows for value in row[key]],
                device=self.device,
                dtype=torch.long,
            )
            labels[name] = pad_sequence(
                list(torch.split(values, lengths)),
                batch_first=True,
                padding_value=-1,
            )
        labels["category"] = self.poi_category_index[
            labels["poi"].clamp_min(0)
        ].masked_fill(labels["poi"] < 0, -1)
        causal_mask, padding_mask = make_transformer_masks(
            lengths, inputs.size(1), self.device
        )
        poi_logits, time_logits, region_logits, category_logits = (
            self.sequence_model(inputs.float(), causal_mask, padding_mask)
        )
        adjusted_logits = continuous_spatiotemporal_fusion(
            poi_logits,
            region_logits,
            time_logits,
            users,
            lengths,
            self.poi_region_index,
            self.user_time_poi,
            region_weight=REGION_WEIGHT,
            time_weight=TIME_WEIGHT,
            smoothing=PRIOR_SMOOTHING,
        )
        return {
            "trajectory_ids": trajectory_ids,
            "users": users,
            "lengths": lengths,
            "labels": labels,
            "poi_logits": poi_logits,
            "time_logits": time_logits,
            "region_logits": region_logits,
            "category_logits": category_logits,
            "adjusted_logits": adjusted_logits,
        }

    def _set_training_mode(self):
        if not self.semantic_stage:
            for module in self.modules.values():
                module.train()
            return
        for module in self.modules.values():
            module.eval()
        for head in (
            self.sequence_model.decoder_poi,
            self.sequence_model.decoder_region,
            self.sequence_model.decoder_time,
            self.sequence_model.decoder_cat,
        ):
            head.train()

    def _set_evaluation_mode(self):
        for module in self.modules.values():
            module.eval()

    def _consistency_loss(self, poi_logits, category_logits, labels):
        poi_probabilities = F.softmax(poi_logits.float(), dim=-1)
        marginal = torch.zeros(
            *poi_probabilities.shape[:-1],
            self.num_categories,
            device=self.device,
        )
        index = self.poi_category_index.view(1, 1, -1).expand_as(
            poi_probabilities
        )
        marginal.scatter_add_(-1, index, poi_probabilities)
        category_probabilities = F.softmax(category_logits.float(), dim=-1)
        midpoint = 0.5 * (marginal + category_probabilities)
        eps = 1e-8
        js = 0.5 * (
            (
                marginal.clamp_min(eps)
                * (
                    marginal.clamp_min(eps).log()
                    - midpoint.clamp_min(eps).log()
                )
            ).sum(dim=-1)
            + (
                category_probabilities.clamp_min(eps)
                * (
                    category_probabilities.clamp_min(eps).log()
                    - midpoint.clamp_min(eps).log()
                )
            ).sum(dim=-1)
        )
        valid = labels >= 0
        return js[valid].mean()

    def _base_loss(self, output):
        labels = output["labels"]
        poi_loss = F.cross_entropy(
            output["adjusted_logits"].transpose(1, 2),
            labels["poi"],
            ignore_index=-1,
        )
        region_loss = F.cross_entropy(
            output["region_logits"].transpose(1, 2),
            labels["region"],
            ignore_index=-1,
        )
        time_loss = F.cross_entropy(
            output["time_logits"].transpose(1, 2),
            labels["time"],
            ignore_index=-1,
        )
        category_loss = F.cross_entropy(
            output["category_logits"].transpose(1, 2),
            labels["category"],
            ignore_index=-1,
        )
        consistency = self._consistency_loss(
            output["poi_logits"],
            output["category_logits"],
            labels["poi"],
        )
        return (
            poi_loss
            + region_loss
            + time_loss
            + category_loss
            + CONSISTENCY_WEIGHT * consistency
        )

    def _semantic_loss(self, output):
        if not self.semantic_stage:
            return output["adjusted_logits"].sum() * 0.0
        score_rows = []
        pair_sets = []
        for row, (trajectory_id, length) in enumerate(zip(
            output["trajectory_ids"], output["lengths"]
        )):
            payload = self.semantic_payloads.get(trajectory_id)
            if payload is None:
                continue
            candidates = payload["candidates"].to(self.device)
            score_rows.append(
                output["adjusted_logits"][row, length - 1, candidates]
            )
            pair_sets.append(payload["pairs"])
        if not score_rows:
            return output["adjusted_logits"].sum() * 0.0
        return ranking_loss(
            torch.stack(score_rows),
            pair_sets,
            batch_size=len(output["trajectory_ids"]),
        )

    def _project(self, output, gate):
        if not self.semantic_stage or self.args.projection_strength == 0:
            return output["adjusted_logits"]
        positions = [
            (row, position)
            for row, (trajectory_id, length) in enumerate(zip(
                output["trajectory_ids"], output["lengths"]
            ))
            for position, is_zero in enumerate(
                zero_support_mask(gate, trajectory_id, length)
            )
            if is_zero
        ]
        if not positions:
            return output["adjusted_logits"]
        rows = torch.tensor(
            [row for row, _ in positions], device=self.device
        )
        steps = torch.tensor(
            [position for _, position in positions], device=self.device
        )
        projected_values = category_marginal_projection(
            output["adjusted_logits"][rows, steps],
            output["category_logits"][rows, steps],
            self.poi_category_index,
            strength=self.args.projection_strength,
            clip=PROJECTION_CLIP,
        )
        projected = output["adjusted_logits"].clone()
        projected[rows, steps] = projected_values
        return projected

    def _zero_support_category_metrics(self, state, output, gate):
        positions = [
            (row, position)
            for row, (trajectory_id, length) in enumerate(zip(
                output["trajectory_ids"], output["lengths"]
            ))
            for position, is_zero in enumerate(
                zero_support_mask(gate, trajectory_id, length)
            )
            if is_zero
        ]
        if not positions:
            return
        rows = torch.tensor(
            [row for row, _ in positions], device=self.device
        )
        steps = torch.tensor(
            [position for _, position in positions], device=self.device
        )
        logits = output["category_logits"][rows, steps].detach()
        targets = output["labels"]["category"][rows, steps]
        target_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        ranks = (logits > target_scores.unsqueeze(1)).sum(dim=1) + 1
        state["count"] += int(len(positions))
        state["mrr"] += float((1.0 / ranks.float()).sum())

    def _build_candidate_pairs(self):
        self._set_evaluation_mode()
        with torch.no_grad():
            for rows in tqdm(
                self.train_reference_loader,
                desc="Building candidate pairs",
            ):
                output = self._forward_batch(rows)
                for row, (trajectory_id, length) in enumerate(zip(
                    output["trajectory_ids"], output["lengths"]
                )):
                    payload = self.semantic_payloads.get(trajectory_id)
                    if payload is None:
                        continue
                    reference = output["adjusted_logits"][row, length - 1].float()
                    candidates = torch.topk(
                        reference,
                        k=min(CANDIDATE_COUNT, self.num_pois),
                        sorted=True,
                    ).indices
                    categories = self.poi_category_index[candidates]
                    reference_scores = reference[candidates]
                    enhanced_scores = (
                        reference_scores + payload["residual"][categories]
                    )
                    payload["candidates"] = candidates.detach().cpu()
                    payload["pairs"] = reversal_pairs(
                        reference_scores,
                        enhanced_scores,
                        categories,
                        threshold=REVERSAL_THRESHOLD,
                    ).detach().cpu()
        ready = [
            payload for payload in self.semantic_payloads.values()
            if "pairs" in payload
        ]
        if not ready:
            raise RuntimeError("no candidate pairs could be constructed")
        self.candidate_summary = {
            "candidate_count": min(CANDIDATE_COUNT, self.num_pois),
            "reversal_threshold": REVERSAL_THRESHOLD,
            "instances": len(ready),
            "instances_with_pairs": sum(payload["pairs"].numel() > 0 for payload in ready),
            "mean_pair_count": float(np.mean([
                payload["pairs"].size(0) for payload in ready
            ])),
        }
        print(json.dumps({"candidate_pairs": self.candidate_summary}, sort_keys=True))

    def _build_optimizer(self):
        parameters = [
            parameter
            for module in self.modules.values()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("no trainable parameters")
        self.optimizer = optim.Adam(
            parameters,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.args.lr_scheduler_factor,
            patience=self.args.patience,
            min_lr=self.args.min_lr,
        )
        print(json.dumps({
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameters
            )
        }))

    def train_epoch(self):
        self._set_training_mode()
        total_losses = []
        semantic_losses = []
        for rows in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            output = self._forward_batch(rows)
            base = self._base_loss(output)
            semantic = self._semantic_loss(output)
            loss = base + self.args.semantic_weight * semantic
            loss.backward()
            self.optimizer.step()
            total_losses.append(float(loss.detach()))
            semantic_losses.append(float(semantic.detach()))
        return float(np.mean(total_losses)), float(np.mean(semantic_losses))

    @torch.no_grad()
    def evaluate(self, loader, gate):
        self._set_evaluation_mode()
        metrics = metric_state()
        category_metrics = {"count": 0, "mrr": 0.0}
        losses = []
        for rows in loader:
            output = self._forward_batch(rows)
            losses.append(float(self._base_loss(output)))
            projected = self._project(output, gate)
            update_metrics(
                metrics,
                projected,
                output["labels"]["poi"],
                output["lengths"],
            )
            self._zero_support_category_metrics(category_metrics, output, gate)
        category_count = category_metrics["count"]
        category_summary = {
            "mrr": category_metrics["mrr"] / max(category_count, 1),
            "count": category_count,
        }
        return (
            summarize_metrics(metrics),
            category_summary,
            float(np.mean(losses)),
        )

    def _capture_modules(self):
        return {
            name: {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
            for name, module in self.modules.items()
        }

    def _checkpoint(self, epoch, validation_metrics, validation_category_metrics):
        checkpoint = {
            "stage": "semantic_enhancement" if self.semantic_stage else "sequence_encoder",
            "experiment": self.args.experiment,
            "epoch": epoch,
            **self._capture_modules(),
            "poi_id_to_index": self.poi_id_to_index,
            "user_id_to_index": self.user_id_to_index,
            "time_id_to_index": self.time_id_to_index,
            "poi_to_region": self.poi_to_region,
            "support_counts": dict(self.support_counts),
            "category_names": self.categories,
            "category_mapping_sha256": self.taxonomy_fingerprint,
            "model_config": self.model_config(),
            "validation_metrics": validation_metrics,
            "validation_zero_support_category_metrics": validation_category_metrics,
            "reference_checkpoint": self.reference_summary or None,
            "semantic_config": (
                {
                    "weight": self.args.semantic_weight,
                    "residual_clip": RESIDUAL_CLIP,
                    "candidate_count": CANDIDATE_COUNT,
                    "reversal_threshold": REVERSAL_THRESHOLD,
                    "projection_strength": self.args.projection_strength,
                    "projection_clip": PROJECTION_CLIP,
                    "teacher_model": TEACHER_MODEL,
                    "candidate_summary": getattr(self, "candidate_summary", None),
                }
                if self.semantic_stage
                else None
            ),
            "user_poi": self.user_poi.detach().cpu(),
            "user_time_poi": self.user_time_poi.detach().cpu(),
            "poi_edges": self.poi_edges.detach().cpu(),
            "region_edges": self.region_edges.detach().cpu(),
        }
        return checkpoint

    def _save_checkpoint(self, checkpoint):
        if not self.args.save_model:
            return
        path = Path(self.args.save_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def run(self):
        best_checkpoint = None
        best_mrr = -float("inf")
        stale_epochs = 0
        for epoch in range(self.args.epochs):
            train_loss, semantic_loss = self.train_epoch()
            validation, validation_category, validation_loss = self.evaluate(
                self.val_loader, self.val_gate
            )
            self.scheduler.step(validation["mrr"])
            if validation["mrr"] > best_mrr:
                best_mrr = validation["mrr"]
                stale_epochs = 0
                best_checkpoint = self._checkpoint(
                    epoch, validation, validation_category
                )
                self._save_checkpoint(best_checkpoint)
            else:
                stale_epochs += 1
            print(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "semantic_loss": semantic_loss,
                "validation_loss": validation_loss,
                "validation_mrr": validation["mrr"],
                "best_validation_mrr": best_mrr,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }, sort_keys=True))
            if stale_epochs >= self.args.early_stop:
                break
        if best_checkpoint is None:
            raise RuntimeError("training did not produce a checkpoint")
        print(metric_table(
            "Best Validation Metrics",
            best_checkpoint["validation_metrics"],
        ))
        print(category_metric_table(
            "Best Validation Category Metrics (zero-support)",
            best_checkpoint["validation_zero_support_category_metrics"],
        ))
        print("[TrainingComplete]")


def main():
    args = parse_arguments()
    seed_everything(args.seed)
    Experiment(args).run()


if __name__ == "__main__":
    main()
