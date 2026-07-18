from dataclasses import dataclass
from typing import Union

import optuna


@dataclass
class FloatParam:
    low: float
    high: float
    log: bool = False


@dataclass
class IntParam:
    low: int
    high: int


@dataclass
class CategoricalParam:
    choices: list


ParamSpec = Union[FloatParam, IntParam, CategoricalParam]


class SearchParams:
    def __init__(self, **params: ParamSpec):
        self.params: dict[str, ParamSpec] = params

    def suggest(self, trial: optuna.Trial) -> dict:
        """Translate an Optuna trial into a concrete parameter dict."""
        result = {}
        for name, spec in self.params.items():
            if isinstance(spec, FloatParam):
                result[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
            elif isinstance(spec, IntParam):
                result[name] = trial.suggest_int(name, spec.low, spec.high)
            elif isinstance(spec, CategoricalParam):
                result[name] = trial.suggest_categorical(name, spec.choices)
        return result
