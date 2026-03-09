"""Optuna-based strategy parameter optimization."""
import optuna
from typing import Callable, Dict, Any
import numpy as np


def optimize_strategy(
    objective_fn: Callable[[Dict[str, Any]], float],
    param_space: Dict[str, tuple],
    n_trials: int = 50,
    direction: str = "maximize",
) -> Dict[str, Any]:
    """
    Optimize strategy parameters using Optuna.
    objective_fn: receives params dict, returns metric (e.g. Sharpe ratio).
    param_space: {"param_name": (low, high)} for float, or (low, high, "int") for int.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in param_space.items():
            if len(spec) == 3 and spec[2] == "int":
                params[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
            else:
                params[name] = trial.suggest_float(name, spec[0], spec[1])
        return objective_fn(params)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best["best_value"] = study.best_value
    return best
