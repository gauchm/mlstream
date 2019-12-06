from typing import Dict, List
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from torch.utils.data import DataLoader

from ..datasets import LumpedBasin, LumpedH5
from .base_models import LumpedModel
from .nseloss import XGBNSEObjective


class LumpedXGBoost(LumpedModel):
    """Wrapper for XGBoost model on lumped data. """

    def __init__(self, no_static: bool = False, concat_static: bool = True,
                 use_mse: bool = False, run_dir: Path = None, n_jobs: int = 1,
                 seed: int = 0, n_estimators: int = 100,
                 learning_rate: float = 0.01,
                 early_stopping_rounds: int = None,
                 n_cv: int = 5,
                 param_dist: Dict = None,
                 param_search_n_estimators: int = None,
                 param_search_n_iter: int = None,
                 param_search_early_stopping_rounds: int = None,
                 reg_search_param_dist: Dict = None,
                 reg_search_n_iter: int = None,
                 model_path: Path = None):
        if not no_static and not concat_static:
            raise ValueError("XGBoost has to use concat_static.")
        self.model = None
        self.use_mse = use_mse
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_cv = n_cv
        self.param_dist = param_dist
        self.param_search_n_estimators = param_search_n_estimators
        self.param_search_n_iter = param_search_n_iter
        self.param_search_early_stopping_rounds = param_search_early_stopping_rounds
        self.reg_search_param_dist = reg_search_param_dist
        self.reg_search_n_iter = reg_search_n_iter
        self.run_dir = run_dir
        self.n_jobs = n_jobs
        self.seed = seed

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path):
        self.model = pickle.load(open(model_path, 'rb'))

    def train(self, ds: LumpedH5) -> None:
        # Create train/val sets
        loader = DataLoader(ds, batch_size=len(ds), num_workers=self.n_jobs)
        data = next(iter(loader))

        # don't use static variables
        if len(data) == 3:
            x, y, q_stds = data

        # this shouldn't happen since we raise an exception if concat_static is False.
        else:
            raise ValueError("XGBoost has to use concat_static.")

        x = x.reshape(len(x), -1).numpy()
        y = y.reshape(-1).numpy()
        q_stds = q_stds.reshape(-1).numpy()

        # define loss function
        if not self.use_mse:
            # slight hack to enable NSE on XGBoost: replace the target with a unique id
            # so we can figure out the corresponding q_std during the loss calculation.
            y_actual = y.copy()
            y = np.arange(len(y))
            loss = XGBNSEObjective(y, y_actual, q_stds)
            self.objective = loss.nse_objective_xgb_sklearn_api
            self.eval_metric = loss.nse_metric_xgb
            self.scoring = loss.neg_nse_metric_sklearn
        else:
            self.objective = 'reg:squarederror'
            self.eval_metric = 'rmse'
            self.scoring = 'neg_mean_squared_error'

        num_val_samples = int(len(x) * 0.2)
        val_indices = np.random.choice(range(len(x)), size=num_val_samples, replace=False)
        train_indices = np.setdiff1d(range(len(x)), val_indices)

        val = [(x[train_indices], y[train_indices]),
               (x[val_indices], y[val_indices])]

        if self.model is None:
            print("Performing parameter searches.")
            if self.param_dist is None or self.n_cv is None \
                    or self.param_search_n_iter is None \
                    or self.param_search_n_estimators is None \
                    or self.param_search_early_stopping_rounds is None \
                    or self.reg_search_param_dist is None \
                    or self.reg_search_n_iter is None:
                raise ValueError("Need to pass parameter search configuration or load model.")

            best_params = self._param_search(x[train_indices], y[train_indices], val,
                                             self.param_search_n_estimators,
                                             self.param_dist,
                                             self.param_search_n_iter).best_params_
            print(f"Best parameters: {best_params}.")

            # Find regularization parameters in separate search
            for k, v in best_params.items():
                self.reg_search_param_dist[k] = [v]
            model = self._param_search(x[train_indices], y[train_indices], val,
                                       self.param_search_n_estimators,
                                       self.reg_search_param_dist,
                                       self.reg_search_n_iter)
            print(f"Best regularization parameters: {model.best_params_}.")

            cv_results = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score',
                                                                     ascending=False)
            print(cv_results.filter(regex='param_|mean_test_score|mean_train_score',
                                    axis=1).head())
            print(cv_results.loc[model.best_index_, ['mean_train_score', 'mean_test_score']])

            xgb_params = model.best_params_

        else:
            print('Using model parameters from provided XGBoost model.')
            xgb_params = self.model.get_xgb_params()

        self.model = xgb.XGBRegressor()
        self.model.set_params(**xgb_params)
        self.model.n_estimators = self.n_estimators
        self.model.learning_rate = self.learning_rate
        self.model.objective = self.objective
        self.model.random_state = self.seed
        self.model.n_jobs = self.n_jobs
        print(self.model.get_xgb_params())

        print("Fitting model.")
        self.model.fit(x[train_indices], y[train_indices],
                       eval_set=val, verbose=True,
                       eval_metric=self.eval_metric,
                       early_stopping_rounds=self.early_stopping_rounds)

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
            raise ValueError("XGBoost has to use concat_static or no_static.")

        x = x.reshape(len(x), -1)
        y = y.reshape(-1)

        return self.model.predict(x), y

    def _param_search(self, x_train: np.ndarray, y_train: np.ndarray, eval_set: List,
                      n_estimators: int, param_dist: Dict, n_iter: int) -> Dict:
        """Performs a cross-validated random parameter search.

        Parameters
        ----------
        x_train : np.ndarray
            Training input
        y_train : np.ndarray
            Training ground truth
        eval_set : List
            List of evaluation sets to report metrics on
        n_estimators : int
            Number of trees to train
        param_dist : Dict
            Search space of parameter distributions
        n_iter : int
            Number of random parameter samples to test

        Returns
        -------
        RandomizedSearchCV
            Fitted random search instance (with ``refit=False``)
        """
        model = xgb.XGBRegressor(n_estimators=n_estimators, objective=self.objective,
                                 n_jobs=1, random_state=self.seed)
        model = RandomizedSearchCV(model, param_dist, n_iter=n_iter,
                                   cv=self.n_cv, return_train_score=True,
                                   scoring=self.scoring, n_jobs=self.n_jobs,
                                   random_state=self.seed, refit=False,
                                   verbose=5, error_score='raise')
        model.fit(x_train, y_train, eval_set=eval_set, eval_metric=self.eval_metric,
                  early_stopping_rounds=self.param_search_early_stopping_rounds, verbose=False)

        return model
