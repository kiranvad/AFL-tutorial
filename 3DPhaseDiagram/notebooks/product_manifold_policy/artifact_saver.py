import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    """Convert supported values into numpy arrays."""
    if isinstance(value, np.ndarray):
        arr = value
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif isinstance(value, (int, float, bool, str)):
        arr = np.asarray(value)
    else:
        raise TypeError(f"Unsupported value type for artifact serialization: {type(value)}")

    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _flatten(prefix: str, data: Mapping[str, Any], store: Dict[str, np.ndarray]) -> None:
    """Flatten nested dictionaries into a single dict with dot notation keys."""
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            _flatten(full_key, value, store)
        else:
            store[full_key] = _to_numpy(value)


@dataclass
class PosteriorArtifacts:
    labels: torch.Tensor
    entropy: torch.Tensor
    accuracy: float


@dataclass
class AcquisitionArtifacts:
    scores: np.ndarray
    xy_star: torch.Tensor
    trajectory: torch.Tensor


@dataclass
class FunctionOptimizationArtifacts:
    trajectory: torch.Tensor
    profile_time: np.ndarray
    profile_temperature: np.ndarray


@dataclass
class ActiveLearningArtifacts:
    train_X: torch.Tensor
    train_Y: torch.Tensor
    batch_indices: torch.Tensor
    xy_star: torch.Tensor
    proposed_temperatures: torch.Tensor


class ExperimentArtifactSaver:
    """Helper for persisting all data needed to recreate generated figures."""

    def __init__(
        self,
        root_dir: Path,
        config: Dict[str, Any],
        bounds: torch.Tensor,
        grid_xy: torch.Tensor,
        grid_t: torch.Tensor,
        grid_xyz: torch.Tensor,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.figure_dir = self.root_dir / "figures"
        self.artifact_dir = self.root_dir / "artifacts"
        self.models_dir = self.artifact_dir / "models"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_path = self.artifact_dir / "metadata.json"
        self._grid_path = self.artifact_dir / "static_grids.npz"
        self._geodesic_path = self.artifact_dir / "geodesic_scores.json"

        if not self._metadata_path.exists():
            metadata = {
                "config": config,
                "bounds": bounds.detach().cpu().tolist(),
                "grid": config.get("grid", {}),
            }
            self._metadata_path.write_text(json.dumps(metadata, indent=2))

        if not self._grid_path.exists():
            np.savez_compressed(
                self._grid_path,
                grid_xy=_to_numpy(grid_xy),
                grid_t=_to_numpy(grid_t),
                grid_xyz=_to_numpy(grid_xyz),
            )

    @property
    def geodesic_history_path(self) -> Path:
        return self._geodesic_path

    def save_geodesic_history(self, source: Path) -> None:
        shutil.copy2(source, self._geodesic_path)

    def iteration_path(self, iteration: int) -> Path:
        return self.artifact_dir / f"iter_{iteration:04d}.npz"

    def figure_path(self, iteration: int) -> Path:
        return self.figure_dir / f"{iteration}.png"

    def gp_model_path(self, iteration: int) -> Path:
        return self.models_dir / f"iter_{iteration:04d}_gp.pt"

    def save_gp_state(self, iteration: int, model: torch.nn.Module, likelihood: torch.nn.Module) -> Path:
        payload = {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
        }
        path = self.gp_model_path(iteration)
        torch.save(payload, path)
        return path

    def log_iteration(
        self,
        iteration: int,
        posterior: PosteriorArtifacts,
        acquisition: AcquisitionArtifacts,
        function_opt: FunctionOptimizationArtifacts,
        active_learning: ActiveLearningArtifacts,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, np.ndarray] = {}
        _flatten("posterior", asdict(posterior), payload)
        _flatten("acquisition_xy", asdict(acquisition), payload)
        _flatten("function_opt", asdict(function_opt), payload)
        _flatten("active_learning", asdict(active_learning), payload)

        if extras:
            _flatten("extras", extras, payload)

        payload["iteration"] = np.asarray(iteration, dtype=np.int32)
        np.savez_compressed(self.iteration_path(iteration), **payload)
