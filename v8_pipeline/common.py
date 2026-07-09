from __future__ import annotations

import json
import math
import re
import shutil
import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASETS = ["Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small"]


def ensure_large_csv_field_limit(min_limit: int = 1024 * 1024 * 1024) -> None:
    """Allow long sequence columns in prepared KT CSV files."""

    current = csv.field_size_limit()
    if current >= min_limit:
        return
    limit = min_limit
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def kg4er_new_root() -> Path:
    return Path(__file__).resolve().parents[1]


def er_root() -> Path:
    return kg4er_new_root().parent


def project_root() -> Path:
    return er_root().parent


def default_data_root() -> Path:
    return er_root() / "KG4ER" / "data"


def default_educdm_root() -> Path:
    return project_root() / "EduCDM_MIRT_noQ_export_modified" / "EduCDM-main"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def locate_dataset_dir(dataset: str, data_root: Path | None = None) -> Path:
    root = data_root or default_data_root()
    path = root / dataset
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def locate_graph_dir(dataset_dir: Path) -> Path:
    prepared = dataset_dir / "prepared_for_kt"
    if (prepared / "entities.dict").exists():
        return prepared
    if (dataset_dir / "entities.dict").exists():
        return dataset_dir
    raise FileNotFoundError(f"Cannot locate entities.dict under {dataset_dir}")


def locate_sequence_file(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "prepared_for_kt" / "sequence_interactions.csv",
        dataset_dir / "processed" / "sequence_interactions.csv",
        dataset_dir / "sequence_interactions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate sequence_interactions.csv under {dataset_dir}")


def locate_q_file(dataset_dir: Path, graph_dir: Path | None = None) -> Path:
    candidates = []
    if graph_dir is not None:
        candidates.append(graph_dir / "Q.txt")
    candidates.extend([dataset_dir / "prepared_for_kt" / "Q.txt", dataset_dir / "Q.txt", dataset_dir / "processed" / "Q.txt"])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate Q.txt under {dataset_dir}")


def locate_semantic_feature_dir(dataset_dir: Path, graph_dir: Path | None = None) -> Path | None:
    candidates = []
    if graph_dir is not None:
        candidates.append(graph_dir / "semantic_kg_features")
    candidates.extend(
        [
            dataset_dir / "semantic_kg_features",
            dataset_dir / "prepared_for_kt" / "semantic_kg_features",
            dataset_dir / "processed" / "semantic_kg_features",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def read_entity_dict(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split("\t")
            result[name] = int(idx)
    return result


def count_entities_by_prefix(entity_dict: dict[str, int], prefix: str) -> int:
    return sum(1 for name in entity_dict if name.startswith(prefix))


def graph_learner_count(graph_dir: Path) -> int:
    return count_entities_by_prefix(read_entity_dict(graph_dir / "entities.dict"), "uid")


def _identifier_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    candidates = [text]
    try:
        number = float(text)
        if number.is_integer():
            candidates.append(str(int(number)))
    except ValueError:
        pass
    return list(dict.fromkeys(candidates))


def _raw_to_fit_uid(input_dir: Path) -> dict[str, int]:
    manifest_path = input_dir / "mirt_input_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MIRT input manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    reverse = manifest.get("maps", {}).get("user_id_to_raw", {})
    return {str(raw): int(fit_uid) for fit_uid, raw in reverse.items()}


def _find_fit_uid(raw_to_fit: dict[str, int], candidates: Iterable[Any]) -> int | None:
    for candidate in candidates:
        for key in _identifier_candidates(candidate):
            if key in raw_to_fit:
                return int(raw_to_fit[key])
    return None


def graph_learner_fit_indices(dataset: str, dataset_dir: Path, graph_dir: Path, input_dir: Path) -> tuple[list[int], dict[str, Any]]:
    """Map ER graph learners uid0..uidN to fitted MIRT user indices.

    MIRT is estimated on `sequence_interactions.csv`, which may include training
    learners as well as test learners. ER recommendation files, however, only
    use the learners present in the graph directory. This function keeps the
    exported V8 mastery/features aligned with `entities.dict`.
    """

    learner_count = graph_learner_count(graph_dir)
    raw_to_fit = _raw_to_fit_uid(input_dir)
    fit_indices: list[int] = []
    missing: list[dict[str, Any]] = []
    source = "graph_uid_sequence"

    test_sequences = graph_dir / "test_sequences.csv"
    if test_sequences.exists():
        source = "test_sequences.csv"
        ensure_large_csv_field_limit()
        with test_sequences.open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        if len(rows) < learner_count:
            missing.append({"reason": "test_sequences_shorter_than_entities", "rows": len(rows), "learner_count": learner_count})
        for idx in range(learner_count):
            row = rows[idx] if idx < len(rows) else {}
            candidates: list[Any] = []
            for key in ["uid", "user_id", "original_uid", "raw_uid"]:
                if key in row:
                    candidates.append(row.get(key))
            if "uid" in row:
                candidates.extend([f"test_{row.get('uid')}", f"uid{row.get('uid')}"])
            fit_uid = _find_fit_uid(raw_to_fit, candidates)
            if fit_uid is None:
                missing.append({"entity_id": f"uid{idx}", "candidates": [str(x) for x in candidates if x is not None]})
            else:
                fit_indices.append(fit_uid)
    else:
        for idx in range(learner_count):
            candidates = [str(idx), idx, f"uid{idx}", f"test_{idx}"]
            fit_uid = _find_fit_uid(raw_to_fit, candidates)
            if fit_uid is None:
                missing.append({"entity_id": f"uid{idx}", "candidates": [str(x) for x in candidates]})
            else:
                fit_indices.append(fit_uid)

    if missing or len(fit_indices) != learner_count:
        raise ValueError(
            "Cannot align ER graph learners to MIRT user ids. "
            f"dataset={dataset}, source={source}, learner_count={learner_count}, "
            f"matched={len(fit_indices)}, missing_examples={missing[:5]}"
        )

    return fit_indices, {
        "dataset": dataset,
        "source": source,
        "learner_count": learner_count,
        "mirt_user_count": len(raw_to_fit),
        "matched_count": len(fit_indices),
        "first_fit_indices": fit_indices[:10],
        "notes": "Rows are ordered as uid0..uidN in entities.dict and selected from the full MIRT user parameter matrix.",
    }


def read_q_matrix(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append([int(float(x)) for x in re.split(r"[,\s]+", line) if x != ""])
    if not rows:
        raise ValueError(f"Empty Q matrix: {path}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Ragged Q matrix: {path}")
    return np.asarray(rows, dtype=np.int64)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def minmax(values: np.ndarray, default: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, default, dtype=np.float64)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, default, dtype=np.float64)
    return (values - lo) / (hi - lo)


def parse_concepts(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value)
    return [int(x) for x in re.findall(r"-?\d+", text)]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
