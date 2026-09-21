"""Utilities for generating a random mice biomedical dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data") / "mice_dataset.csv"
N_ROWS = 140


def _balanced_labels(rng: np.random.Generator, first: str, second: str, size: int) -> np.ndarray:
    labels = np.array([first] * (size // 2) + [second] * (size - size // 2))
    rng.shuffle(labels)
    return labels


def _clip_corr_values(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(values, lower, upper)


def _resolve_output_path() -> Path:
    output_path = Path.cwd() / OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _introduce_cols_missing_values(rng: np.random.Generator, dataset: pd.DataFrame) -> None:
    protein_columns = [column for column in dataset.columns if column.startswith("Protein")]
    selected_column = rng.choice(protein_columns)
    missing_count = max(1, int(np.ceil(0.8 * len(dataset))))
    missing_rows = rng.choice(dataset.index, size=missing_count, replace=False)
    dataset.loc[missing_rows, selected_column] = np.nan


def _introduce_row_missing_values(rng: np.random.Generator, dataset: pd.DataFrame) -> None:
    selected_rows = rng.choice(dataset.index, size=3, replace=False)
    candidate_columns = [column for column in dataset.columns if column != "ID"]
    missing_columns_count = max(1, int(np.ceil(0.8 * len(candidate_columns))))

    for row_index in selected_rows:
        missing_columns = rng.choice(candidate_columns, size=missing_columns_count, replace=False)
        dataset.loc[row_index, missing_columns] = np.nan


def _random_transpose(rng: np.random.Generator, dataset: pd.DataFrame) -> pd.DataFrame:
    if rng.random() < 0.5:
        return dataset.transpose()
    return dataset

def save_random_mice_dataset() -> pd.DataFrame:
    """Create, save, and return a random mice biomedical dataset."""

    rng = np.random.default_rng()

    ids = np.array([f"Mice{index:03d}" for index in range(1, N_ROWS + 1)])
    sex = _balanced_labels(rng, "F", "M", N_ROWS)

    group = np.empty(N_ROWS, dtype=object)
    for current_sex in ("F", "M"):
        sex_mask = sex == current_sex
        group[sex_mask] = _balanced_labels(rng, "Control", "Treatment", sex_mask.sum())

    age = rng.integers(26, 35, size=N_ROWS)

    sex_signal = (sex == "M").astype(float)
    treatment_signal = (group == "Treatment").astype(float)

    for _ in range(1000):
        protein1_raw = 100 + 18 * sex_signal + rng.normal(0, 12, size=N_ROWS) #
        protein2_raw = 160 + 8 * (age - 30) + rng.normal(0, 20, size=N_ROWS)
        protein3_raw = 25 + 8 * treatment_signal + rng.normal(0, 4.5, size=N_ROWS)

        protein1 = _clip_corr_values(protein1_raw, 10, 200)
        protein2 = _clip_corr_values(protein2_raw, 50, 300)
        protein3 = _clip_corr_values(protein3_raw, 10, 50)

        corr1 = pd.Series(protein1).corr(pd.Series(sex_signal))
        corr2 = pd.Series(protein2).corr(pd.Series(age))
        corr3 = pd.Series(protein3).corr(pd.Series(treatment_signal))

        if 0.6 <= corr1 <= 0.8 and 0.6 <= corr2 <= 0.8 and 0.6 <= corr3 <= 0.8:
            break
    else:
        raise RuntimeError("Could not generate the data. Try again.")

    weight = np.where(
        sex == "F",
        rng.normal(21, 2, size=N_ROWS),
        rng.normal(28, 2, size=N_ROWS),
    )
    protein4 = rng.uniform(10, 1000, size=N_ROWS)
    protein5 = 1.54 * protein1 + 0.37 * protein2 + rng.normal(0, 5, size=N_ROWS)
    protein6 = 9.2 * protein3 + 1.5 * age + rng.normal(0, 2, size=N_ROWS)

    dataset = pd.DataFrame(
        {
            "ID": ids,
            "Sex": sex,
            "Group": group,
            "Weight(g)": weight,
            "Age(weeks)": age,
            "Protein1": protein1,
            "Protein2": protein2,
            "Protein3": protein3,
            "Protein4": protein4,
            "Protein5": protein5,
            "Protein6": protein6,
        }
    )

    _introduce_cols_missing_values(rng, dataset)
    _introduce_row_missing_values(rng, dataset)
    dataset = _random_transpose(rng, dataset)
    dataset.to_csv(_resolve_output_path(), index=False)
    return dataset


if __name__ == "__main__":
    save_random_mice_dataset()