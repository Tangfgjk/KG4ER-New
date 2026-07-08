from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASETS = ["Eedi", "algebra2005", "assist2009-sub", "statics2011", "XES3G5M-sub-small"]


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

