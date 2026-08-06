from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


RSGP_STRUCTURAL_PRIOR_VERSION = 1


def metadata_order_hash(
    class_names: Sequence[object],
    predicate_names: Sequence[object],
) -> str:
    payload = {
        "class_names": [str(name) for name in class_names],
        "predicate_names": [str(name) for name in predicate_names],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_hash(payload: Dict[str, Any]) -> str:
    content = dict(payload)
    content.pop("payload_hash", None)
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return ranks


def _box_state(record: Dict[str, Any], box_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(record["boxes"], dtype=np.float64)
    if boxes.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    if box_mode == "obb":
        return boxes[:, :2], np.maximum(boxes[:, 2:4], 1e-6), np.deg2rad(boxes[:, 4])
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:4])
    sizes = np.maximum(boxes[:, 2:4] - boxes[:, :2], 1e-6)
    return centers, sizes, np.zeros((boxes.shape[0],), dtype=np.float64)


def _containment_rates(
    centers: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    *,
    block_size: int = 64,
) -> np.ndarray:
    """Return the fraction of other entity centers inside each OBB.

    The computation is block-wise so the temporary allocation is bounded by
    ``block_size x num_entities`` rather than materializing all geometric
    intermediates at once.
    """

    num_entities = int(centers.shape[0])
    rates = np.zeros((num_entities,), dtype=np.float64)
    if num_entities <= 1:
        return rates
    indices = np.arange(num_entities)
    for start in range(0, num_entities, max(int(block_size), 1)):
        end = min(start + max(int(block_size), 1), num_entities)
        delta = centers[None, :, :] - centers[start:end, None, :]
        cos = np.cos(angles[start:end])[:, None]
        sin = np.sin(angles[start:end])[:, None]
        local_x = delta[:, :, 0] * cos + delta[:, :, 1] * sin
        local_y = -delta[:, :, 0] * sin + delta[:, :, 1] * cos
        inside = (
            (np.abs(local_x) <= sizes[start:end, None, 0] * 0.5)
            & (np.abs(local_y) <= sizes[start:end, None, 1] * 0.5)
        )
        inside[np.arange(end - start), indices[start:end]] = False
        rates[start:end] = inside.sum(axis=1, dtype=np.float64) / float(num_entities - 1)
    return rates


def _unique_relation_degree(relations: np.ndarray, num_entities: int) -> np.ndarray:
    degree = np.zeros((num_entities,), dtype=np.float64)
    if num_entities <= 1 or relations.size == 0:
        return degree
    pairs = np.asarray(relations[:, :2], dtype=np.int64)
    valid = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < num_entities)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < num_entities)
        & (pairs[:, 0] != pairs[:, 1])
    )
    pairs = pairs[valid]
    if pairs.size == 0:
        return degree
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    directed = np.unique(directed, axis=0)
    counts = np.bincount(directed[:, 0], minlength=num_entities).astype(np.float64)
    return np.log1p(counts) / max(math.log1p(num_entities), 1e-6)


def build_rsgp_structural_prior(
    dataset,
    *,
    split: str = "train",
    rarity_beta: float = 0.5,
    containment_block_size: int = 64,
) -> Dict[str, Any]:
    """Build category-name-agnostic structural profiles from one train split."""

    if str(split).lower() != "train":
        raise ValueError("RSGP structural priors must be built from split='train'.")
    if getattr(dataset, "split", "train").lower() != "train":
        raise ValueError(
            f"Expected a train dataset, got split={getattr(dataset, 'split', None)!r}."
        )
    if rarity_beta < 0:
        raise ValueError("rarity_beta must be non-negative.")

    metadata = dataset.metadata
    num_classes = int(metadata.num_classes)
    num_predicates = int(metadata.num_predicates)
    class_names = [
        str(metadata.categories.get(index, index)) for index in range(num_classes)
    ]
    predicate_names = [
        str(metadata.predicates.get(index, index)) for index in range(num_predicates)
    ]
    role_sums = np.zeros((num_classes, 3), dtype=np.float64)
    class_counts = np.zeros((num_classes,), dtype=np.int64)
    predicate_frequency = np.zeros((num_predicates,), dtype=np.int64)
    predicate_support = np.zeros(
        (num_classes, num_classes, num_predicates),
        dtype=np.bool_,
    )
    dataset_hasher = hashlib.sha256()
    box_mode = str(getattr(dataset, "box_mode", metadata.box_mode)).lower()

    for record in dataset.records:
        labels = np.asarray(record.get("labels", []), dtype=np.int64).reshape(-1)
        relations = np.asarray(record.get("relations", []), dtype=np.int64)
        if relations.size == 0:
            relations = np.zeros((0, 3), dtype=np.int64)
        else:
            relations = relations.reshape(-1, 3)
        centers, sizes, angles = _box_state(record, box_mode)
        num_entities = int(labels.size)
        if centers.shape[0] != num_entities:
            raise ValueError("Record box/label count mismatch while building RSGP prior.")

        area_rank = _rank_percentiles(sizes[:, 0] * sizes[:, 1])
        containment = _containment_rates(
            centers,
            sizes,
            angles,
            block_size=containment_block_size,
        )
        elongation = 1.0 - (
            np.minimum(sizes[:, 0], sizes[:, 1])
            / np.maximum(sizes[:, 0], sizes[:, 1]).clip(min=1e-6)
        )
        connectivity = _unique_relation_degree(relations, num_entities)
        region = 0.5 * area_rank + 0.5 * containment

        valid_labels = (labels > 0) & (labels < num_classes)
        for role_index, values in enumerate((region, elongation, connectivity)):
            np.add.at(role_sums[:, role_index], labels[valid_labels], values[valid_labels])
        np.add.at(class_counts, labels[valid_labels], 1)

        for subj, obj, predicate in relations.tolist():
            subj, obj, predicate = int(subj), int(obj), int(predicate)
            if not (
                0 <= subj < num_entities
                and 0 <= obj < num_entities
                and 0 < predicate < num_predicates
            ):
                continue
            subj_label, obj_label = int(labels[subj]), int(labels[obj])
            if not (0 <= subj_label < num_classes and 0 <= obj_label < num_classes):
                continue
            predicate_frequency[predicate] += 1
            predicate_support[subj_label, obj_label, predicate] = True

        dataset_hasher.update(
            np.asarray(
                [
                    int(record.get("image_index", 0)),
                    num_entities,
                    int(relations.shape[0]),
                ],
                dtype=np.int64,
            ).tobytes()
        )
        dataset_hasher.update(np.ascontiguousarray(labels).tobytes())
        dataset_hasher.update(
            np.ascontiguousarray(np.asarray(record["boxes"], dtype=np.float32)).tobytes()
        )
        dataset_hasher.update(np.ascontiguousarray(relations).tobytes())

    profiles = np.zeros_like(role_sums)
    populated = class_counts > 0
    profiles[populated] = role_sums[populated] / class_counts[populated, None]
    profiles = np.clip(profiles, 0.0, 1.0)

    rarity = np.zeros((num_predicates,), dtype=np.float64)
    active_predicates = predicate_frequency > 0
    if active_predicates.any():
        active_rarity = np.power(
            predicate_frequency[active_predicates].astype(np.float64) + 1e-12,
            -float(rarity_beta),
        )
        rarity_max = float(active_rarity.max())
        if rarity_max <= 1e-12:
            active_rarity.fill(1.0)
        else:
            active_rarity = active_rarity / rarity_max
        rarity[active_predicates] = active_rarity
    rarity[0] = 0.0
    rarity_support = (
        predicate_support.astype(np.float64) * rarity.reshape(1, 1, -1)
    ).max(axis=2)

    build_config = {
        "source_split": "train",
        "dataset_name": str(metadata.dataset_name),
        "box_mode": box_mode,
        "rarity_beta": float(rarity_beta),
        "containment_block_size": int(containment_block_size),
        "num_classes": num_classes,
        "num_predicates": num_predicates,
        "metadata_order_hash": metadata_order_hash(class_names, predicate_names),
    }
    dataset_hasher.update(
        json.dumps(
            {
                "dataset_name": str(metadata.dataset_name),
                "box_mode": box_mode,
                "num_images": len(dataset.records),
                "metadata_order_hash": build_config["metadata_order_hash"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    dataset_signature = dataset_hasher.hexdigest()
    payload: Dict[str, Any] = {
        "schema_version": RSGP_STRUCTURAL_PRIOR_VERSION,
        "source_split": "train",
        "dataset_name": str(metadata.dataset_name),
        "box_mode": box_mode,
        "num_classes": num_classes,
        "num_predicates": num_predicates,
        "class_names": class_names,
        "predicate_names": predicate_names,
        "metadata_order_hash": metadata_order_hash(class_names, predicate_names),
        "dataset_signature": dataset_signature,
        "build_config": build_config,
        "config_hash": _mapping_hash(build_config),
        "rarity_beta": float(rarity_beta),
        "num_images": len(dataset.records),
        "class_instance_counts": class_counts.tolist(),
        "class_profiles": {
            "contextual_region": profiles[:, 0].tolist(),
            "directional_alignment": profiles[:, 1].tolist(),
            "relational_connectivity": profiles[:, 2].tolist(),
        },
        "predicate_frequency": predicate_frequency.tolist(),
        "predicate_rarity": rarity.tolist(),
        "rarity_pair_support": rarity_support.tolist(),
    }
    payload["payload_hash"] = _payload_hash(payload)
    return payload


def validate_rsgp_structural_prior(
    payload: Dict[str, Any],
    *,
    class_names: Sequence[object],
    predicate_names: Sequence[object],
    expected_hash: str = "",
) -> None:
    version = int(payload.get("schema_version", -1))
    if version != RSGP_STRUCTURAL_PRIOR_VERSION:
        raise RuntimeError(
            "Unsupported RSGP structural prior schema: "
            f"expected={RSGP_STRUCTURAL_PRIOR_VERSION}, actual={version}."
        )
    if str(payload.get("source_split", "")).lower() != "train":
        raise RuntimeError("RSGP structural prior was not generated from the train split.")
    actual_payload_hash = _payload_hash(payload)
    stored_payload_hash = str(payload.get("payload_hash", ""))
    if not stored_payload_hash or stored_payload_hash != actual_payload_hash:
        raise RuntimeError(
            "RSGP structural prior payload hash mismatch: "
            f"stored={stored_payload_hash or '<missing>'}, actual={actual_payload_hash}."
        )
    if expected_hash and stored_payload_hash != str(expected_hash):
        raise RuntimeError(
            "RSGP structural prior does not match RSGP_STRUCTURAL_PRIOR_HASH: "
            f"expected={expected_hash}, actual={stored_payload_hash}."
        )
    build_config = payload.get("build_config")
    stored_config_hash = str(payload.get("config_hash", ""))
    if not isinstance(build_config, dict) or not stored_config_hash:
        raise RuntimeError("RSGP structural prior is missing its build configuration hash.")
    actual_config_hash = _mapping_hash(build_config)
    if stored_config_hash != actual_config_hash:
        raise RuntimeError(
            "RSGP structural prior configuration hash mismatch: "
            f"stored={stored_config_hash}, actual={actual_config_hash}."
        )
    expected_order_hash = metadata_order_hash(class_names, predicate_names)
    actual_order_hash = str(payload.get("metadata_order_hash", ""))
    if actual_order_hash != expected_order_hash:
        raise RuntimeError(
            "RSGP structural prior class/predicate order mismatch: "
            f"expected={expected_order_hash}, actual={actual_order_hash or '<missing>'}."
        )
    if int(payload.get("num_classes", -1)) != len(class_names):
        raise RuntimeError("RSGP structural prior class count mismatch.")
    if int(payload.get("num_predicates", -1)) != len(predicate_names):
        raise RuntimeError("RSGP structural prior predicate count mismatch.")


def load_rsgp_structural_prior(
    path: str | Path,
    *,
    class_names: Sequence[object],
    predicate_names: Sequence[object],
    expected_hash: str = "",
) -> Dict[str, Any]:
    prior_path = Path(path)
    if not prior_path.is_file():
        raise FileNotFoundError(
            f"RSGP structural prior not found: {prior_path}. "
            "Build it with tools/build_rsgp_structural_prior.py."
        )
    try:
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read RSGP structural prior: {prior_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"RSGP structural prior must be a JSON object: {prior_path}")
    validate_rsgp_structural_prior(
        payload,
        class_names=class_names,
        predicate_names=predicate_names,
        expected_hash=expected_hash,
    )
    return payload


def write_rsgp_structural_prior(payload: Dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
