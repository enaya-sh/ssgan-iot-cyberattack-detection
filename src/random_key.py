from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContinuousSpec:
    low: float
    high: float


@dataclass(frozen=True)
class IntegerSpec:
    low: int
    high: int


@dataclass(frozen=True)
class CategoricalSpec:
    values: tuple[Any, ...]


class RandomKeyDecoder:
    """
    Mixed-variable Random Key decoder.

    Continuous/integer variables use one key in [0,1].
    Categorical variables use len(values) keys; the index of the largest
    key selects the category.

    This follows the thesis description at a high level. The thesis does
    not provide implementation-level tie-breaking details, so this module
    is a transparent reference implementation rather than a claim of exact
    source-code reproduction.
    """

    def __init__(self, schema: dict[str, object]):
        self.schema = schema
        self.slices = {}
        start = 0
        for name, spec in schema.items():
            width = (
                len(spec.values)
                if isinstance(spec, CategoricalSpec)
                else 1
            )
            self.slices[name] = slice(start, start + width)
            start += width
        self.dimension = start

    def decode(self, key: np.ndarray) -> dict[str, object]:
        if len(key) != self.dimension:
            raise ValueError(
                f"Expected key dimension {self.dimension}, got {len(key)}"
            )

        out = {}
        for name, spec in self.schema.items():
            segment = key[self.slices[name]]

            if isinstance(spec, ContinuousSpec):
                out[name] = spec.low + float(segment[0]) * (
                    spec.high - spec.low
                )

            elif isinstance(spec, IntegerSpec):
                raw = spec.low + float(segment[0]) * (
                    spec.high - spec.low
                )
                out[name] = int(round(raw))

            elif isinstance(spec, CategoricalSpec):
                out[name] = spec.values[int(np.argmax(segment))]

            else:
                raise TypeError(f"Unsupported specification: {spec!r}")

        return out
