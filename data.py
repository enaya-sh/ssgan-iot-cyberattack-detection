from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class IoTDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]
        return self.data[idx], self.labels[idx]


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_unlabeled: np.ndarray
    num_classes: int


def load_generic_csv(
    path: str,
    label_column: str | None = None,
    labeled_fraction: float = 0.20,
    random_state: int = 42,
) -> SplitData:
    """
    Generic portfolio loader.

    Important:
    The thesis describes dataset-specific cleaning, encoding, balancing,
    normalization and feature handling for NSL-KDD, MAWI and CICIoT2023.
    This generic loader does not claim to reproduce those dataset-specific
    pipelines. It expects an already numeric, cleaned CSV.
    """
    df = pd.read_csv(path)

    if label_column is None:
        label_column = df.columns[-1]

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    feature_df = df.drop(columns=[label_column])
    non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Generic loader expects numeric features. "
            f"Non-numeric columns: {non_numeric}"
        )

    if feature_df.isnull().any().any() or df[label_column].isnull().any():
        raise ValueError(
            "Generic loader expects a cleaned dataset without missing values."
        )

    y_raw = df[label_column].to_numpy()
    x = feature_df.to_numpy(dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(
        x,
        y,
        train_size=labeled_fraction,
        stratify=y,
        random_state=random_state,
    )

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x_labeled,
        y_labeled,
        test_size=0.40,
        stratify=y_labeled,
        random_state=random_state,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=random_state,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    x_unlabeled = scaler.transform(x_unlabeled)

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        x_unlabeled=x_unlabeled,
        num_classes=len(encoder.classes_),
    )


def make_labeled_loader(x, y, batch_size: int, shuffle: bool):
    ds = IoTDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_unlabeled_loader(x, batch_size: int, shuffle: bool = True):
    ds = IoTDataset(torch.tensor(x, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
