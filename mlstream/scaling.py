from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

from .datautils import (load_forcings_lumped,
                        load_discharge,
                        load_static_attributes)


class Scaler:

    def __init__(self):
        self.scalers = {}

    def normalize(self, feature: np.ndarray) -> np.ndarray:
        return (feature - self.scalers["mean"]) / self.scalers["std"]

    def rescale(self, feature: np.ndarray) -> np.ndarray:
        return (feature * self.scalers["std"]) + self.scalers["mean"]


class InputScaler(Scaler):

    def __init__(self, data_root: Path, basins: List,
                 start_date: pd.Timestamp, end_date: pd.Timestamp):
        super().__init__()

        all_forcings = pd.DataFrame()
        print("Loading forcings for input scaler.")
        basin_forcings = load_forcings_lumped(data_root, basins)
        for basin, forcing in basin_forcings.items():
            all_forcings = all_forcings.append(forcing.loc[start_date:end_date])
        self.scalers["mean"] = all_forcings.mean(axis=0).values

        stds = all_forcings.std(axis=0).values
        stds[stds == 0] = 1  # avoid divide-by-zero
        self.scalers["std"] = stds


class OutputScaler(Scaler):

    def __init__(self, data_root: Path, basins: List,
                 start_date: pd.Timestamp, end_date: pd.Timestamp):
        super().__init__()

        print("Loading streamflow for output scaler.")
        all_outputs = load_discharge(data_root, basins)
        all_outputs = all_outputs[(all_outputs['date'] >= start_date)
                                  & (all_outputs['date'] <= end_date)]
        self.scalers["mean"] = all_outputs["qobs"].mean()
        self.scalers["std"] = all_outputs["qobs"].std()


class StaticAttributeScaler(Scaler):

    def __init__(self, db_path: Path, basins: List, variable_name: str):
        super().__init__()

        statics = load_static_attributes(db_path, basins)[variable_name]
        self.scalers["mean"] = statics.mean()

        # avoid divide-by-zero
        self.scalers["std"] = statics.std() if statics.std() != 0 else 1
