from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from ..datasets import LumpedBasin, LumpedH5
from .base_models import LumpedModel


class LumpedLinearRegression(LumpedModel):
    """Linear regression model for lumped data. """

    def __init__(self, num_dynamic_vars, num_static_vars, use_mse: bool = True,
                 no_static: bool = False, concat_static: bool = True,
                 run_dir: Path = None, n_jobs: int = 1):
        if not use_mse:
            print("Linear regression does not support NSE.")
        if not no_static and not concat_static:
            raise ValueError("Linear regression has to use concat_static.")
        self.model = LinearRegression(n_jobs=n_jobs)
        self.run_dir = run_dir
        self.n_jobs = n_jobs

    def load(self, model_file: Path) -> None:
        self.model = pickle.load(open(model_file, 'rb'))

    def train(self, ds: LumpedH5) -> None:
        # Create train/val sets
        loader = DataLoader(ds, batch_size=len(ds), num_workers=self.n_jobs)
        data = next(iter(loader))

        # don't use static variables
        if len(data) == 3:
            x, y, _ = data  # ignore q_stds

        # this shouldn't happen since we raise an exception if concat_static is False.
        else:
            raise ValueError("Linear regression has to use concat_static.")

        x = x.reshape(len(x), -1)
        y = y.reshape(len(y))

        print("Fitting model.")
        self.model.fit(x, y)

        model_path = self.run_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved as {model_path}.")

    def predict(self, ds: LumpedBasin) -> np.ndarray:
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=4)
        data = next(iter(loader))

        if len(data) == 2:
            x, y = data

        # this shouldn't happen since we didn't allow concat_static = False in training.
        else:
            raise ValueError("Linear regression has to use concat_static.")

        x = x.reshape(len(x), -1)
        y = y.reshape(len(y))

        return self.model.predict(x), y
