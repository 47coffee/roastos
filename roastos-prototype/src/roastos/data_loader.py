from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from .models import (
    CoffeeLot,
    RoastFeatures,
    RoastOutcome,
    RoastSession,
    RoastTimeSeriesRow,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ValidationIssue:
    row_number: int
    raw_data: dict
    error: str


@dataclass
class LoadResult(Generic[T]):
    items: list[T]
    errors: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        if self.errors:
            preview = "\n".join(
                f"Row {e.row_number}: {e.error}" for e in self.errors[:10]
            )
            raise ValueError(
                f"Validation failed with {len(self.errors)} error(s):\n{preview}"
            )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _validate_rows(df: pd.DataFrame, model_cls: type[T]) -> LoadResult[T]:
    items: list[T] = []
    errors: list[ValidationIssue] = []

    records = df.to_dict(orient="records")
    for idx, record in enumerate(records, start=2):  # header is line 1
        try:
            item = model_cls.model_validate(record)
            items.append(item)
        except ValidationError as exc:
            errors.append(
                ValidationIssue(
                    row_number=idx,
                    raw_data=record,
                    error=str(exc),
                )
            )
    return LoadResult(items=items, errors=errors)


def load_roast_sessions(path: Path) -> LoadResult[RoastSession]:
    df = _read_csv(path)
    return _validate_rows(df, RoastSession)


def load_roast_timeseries(path: Path) -> LoadResult[RoastTimeSeriesRow]:
    df = _read_csv(path)
    return _validate_rows(df, RoastTimeSeriesRow)


def load_roast_features(path: Path) -> LoadResult[RoastFeatures]:
    df = _read_csv(path)
    return _validate_rows(df, RoastFeatures)


def load_roast_outcomes(path: Path) -> LoadResult[RoastOutcome]:
    df = _read_csv(path)
    return _validate_rows(df, RoastOutcome)


def load_coffee_lots(path: Path) -> LoadResult[CoffeeLot]:
    df = _read_csv(path)
    return _validate_rows(df, CoffeeLot)


@dataclass
class RoastOSDataset:
    sessions: list[RoastSession]
    timeseries: list[RoastTimeSeriesRow]
    features: list[RoastFeatures]
    outcomes: list[RoastOutcome]
    coffee_lots: list[CoffeeLot]


def load_full_dataset(data_dir: str | Path) -> RoastOSDataset:
    data_dir = Path(data_dir)

    sessions_result = load_roast_sessions(data_dir / "roast_sessions.csv")
    timeseries_result = load_roast_timeseries(data_dir / "roast_timeseries.csv")
    features_result = load_roast_features(data_dir / "roast_features.csv")
    outcomes_result = load_roast_outcomes(data_dir / "roast_outcomes.csv")
    coffee_result = load_coffee_lots(data_dir / "coffee_lots.csv")

    sessions_result.raise_if_errors()
    timeseries_result.raise_if_errors()
    features_result.raise_if_errors()
    outcomes_result.raise_if_errors()
    coffee_result.raise_if_errors()

    _validate_cross_references(
        sessions=sessions_result.items,
        timeseries=timeseries_result.items,
        features=features_result.items,
        outcomes=outcomes_result.items,
        coffee_lots=coffee_result.items,
    )

    return RoastOSDataset(
        sessions=sessions_result.items,
        timeseries=timeseries_result.items,
        features=features_result.items,
        outcomes=outcomes_result.items,
        coffee_lots=coffee_result.items,
    )


def _validate_cross_references(
    *,
    sessions: list[RoastSession],
    timeseries: list[RoastTimeSeriesRow],
    features: list[RoastFeatures],
    outcomes: list[RoastOutcome],
    coffee_lots: list[CoffeeLot],
) -> None:
    session_ids = {s.roast_id for s in sessions}
    coffee_ids = {c.coffee_id for c in coffee_lots}

    for session in sessions:
        if session.coffee_id not in coffee_ids:
            raise ValueError(
                f"Session {session.roast_id} references missing coffee_id={session.coffee_id}"
            )

    for row in timeseries:
        if row.roast_id not in session_ids:
            raise ValueError(
                f"Timeseries row references missing roast_id={row.roast_id}"
            )

    for row in features:
        if row.roast_id not in session_ids:
            raise ValueError(
                f"Feature row references missing roast_id={row.roast_id}"
            )

    for row in outcomes:
        if row.roast_id not in session_ids:
            raise ValueError(
                f"Outcome row references missing roast_id={row.roast_id}"
            )

    _validate_timeseries_order(timeseries)


def _validate_timeseries_order(timeseries: list[RoastTimeSeriesRow]) -> None:
    grouped: dict[str, list[RoastTimeSeriesRow]] = {}
    for row in timeseries:
        grouped.setdefault(row.roast_id, []).append(row)

    for roast_id, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: r.t_s)
        for previous, current in zip(rows_sorted, rows_sorted[1:]):
            if current.t_s <= previous.t_s:
                raise ValueError(
                    f"Timeseries for roast {roast_id} is not strictly increasing in t_s"
                )