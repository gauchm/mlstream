import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from .datasets import LumpedBasin, LumpedH5
from .scaling import InputScaler, OutputScaler, StaticAttributeScaler
from .utils import setup_run, prepare_data, nse
from .datautils import load_static_attributes


class Experiment:
    """Main entrypoint for training and prediction. """
    def __init__(self, data_root: Path, is_train: bool, run_dir: Path,
                 start_date: str = None, end_date: str = None,
                 basins: List = None, forcing_attributes: List = None,
                 static_attributes: List = None, seq_length: int = 10,
                 concat_static: bool = False, no_static: bool = False,
                 cache_data: bool = False, n_jobs: int = 1, seed: int = 0,
                 run_metadata: Dict = {}):
        """Initializes the experiment.

        Parameters
        ----------
        data_root : Path
            Path to the base directory of the data set.
        is_train : bool
            If True, will setup folder structure for training.
        run_dir: Path
            Path to store experiment results in.
        start_date : str, optional
            Start date (training start date if ``is_train``, else
            validation start date, ddmmyyyy)
        end_date : str, optional
            End date (training end date if ``is_train``, else
            validation end date, ddmmyyyy)
        basins : List, optional
            List of basins to use during training,
            or basins to predict during prediction.
        forcing_attributes : List, optional
            Names of forcing attributes to use.
        static_attributes : List, optional
            Names of static basin attributes to use.
        seq_length : int, optional
            Length of historical forcings to feed the model. Default 10
        cache_data : bool, optional
            If True, will preload all data in memory for training. Default False
        concat_static : bool, optional
            If True, will concatenate static basin attributes with forcings for model input.
        no_static : bool, optional
            If True, will not use static basin attributes as model input.
        n_jobs : int, optional
            Number of workers to use for training. Default 1
        seed : int, optional
            Seed to use for training. Default 0
        run_metadata : dict, optional
            Optional dictionary of values to store in cfg.json for documentation purpose.
        """
        self.model = None
        self.results = {}

        self.cfg = {
            "data_root": data_root,
            "run_dir": run_dir,
            "start_date": pd.to_datetime(start_date, format='%d%m%Y'),
            "end_date": pd.to_datetime(end_date, format='%d%m%Y'),
            "basins": basins,
            "forcing_attributes": forcing_attributes,
            "static_attributes": static_attributes,
            "seq_length": seq_length,
            "cache_data": cache_data,
            "concat_static": concat_static,
            "no_static": no_static,
            "seed": seed,
            "n_jobs": n_jobs
        }
        self.cfg.update(run_metadata)

        if is_train:
            # create folder structure for this run
            self.cfg = setup_run(self.cfg)

            # prepare data for training
            self.cfg = prepare_data(cfg=self.cfg, basins=basins)

    def set_model(self, model) -> None:
        """Set the model to use in the experiment. """
        self.model = model

    def train(self) -> None:
        """Train model. """
        if self.model is None:
            raise AttributeError("Model is not set.")

        # fix random seeds
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        torch.cuda.manual_seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])

        # prepare PyTorch DataLoader
        ds = LumpedH5(h5_file=self.cfg["train_file"],
                      basins=self.cfg["basins"],
                      db_path=self.cfg["db_path"],
                      concat_static=self.cfg["concat_static"],
                      cache=self.cfg["cache_data"],
                      no_static=self.cfg["no_static"])

        self.model.train(ds)

    def predict(self) -> Dict:
        """Generates predictions with a trained model.

        Returns
        -------
        results : Dict
            Dictionary containing the DataFrame of predictions and observations for each basin.
        """
        with open(self.cfg["run_dir"] / 'cfg.json', 'r') as fp:
            run_cfg = json.load(fp)

        if self.model is None:
            raise AttributeError("Model is not set.")

        # self.cfg["start_date"] contains validation start date,
        # run_cfg["start_date"] the training start date
        run_cfg["start_date"] = pd.to_datetime(run_cfg["start_date"], format='%d%m%Y')
        run_cfg["end_date"] = pd.to_datetime(run_cfg["end_date"], format='%d%m%Y')
        self.cfg["start_date"] = pd.to_datetime(self.cfg["start_date"], format='%d%m%Y')
        self.cfg["end_date"] = pd.to_datetime(self.cfg["end_date"], format='%d%m%Y')

        # create scalers
        input_scalers = InputScaler(self.cfg["data_root"], run_cfg["basins"],
                                    run_cfg["start_date"], run_cfg["end_date"])
        output_scalers = OutputScaler(self.cfg["data_root"], run_cfg["basins"],
                                      run_cfg["start_date"], run_cfg["end_date"])
        static_scalers = {}
        db_path = self.cfg["run_dir"] / "static_attributes.db"
        df = load_static_attributes(db_path, run_cfg["basins"], drop_lat_lon=True)
        for feature in [f for f in df.columns if 'onehot' not in f]:
            static_scalers[feature] = StaticAttributeScaler(db_path, run_cfg["basins"],
                                                            feature)

        # self.cfg["basins"] contains the test basins, run_cfg["basins"] the train basins.
        for basin in tqdm(self.cfg["basins"]):
            ds_test = LumpedBasin(data_root=Path(self.cfg["data_root"]),
                                  basin=basin,
                                  forcing_vars=run_cfg["forcing_attributes"],
                                  dates=[self.cfg["start_date"], self.cfg["end_date"]],
                                  is_train=False,
                                  train_basins=run_cfg["basins"],
                                  seq_length=run_cfg["seq_length"],
                                  with_attributes=not run_cfg["no_static"],
                                  concat_static=run_cfg["concat_static"],
                                  db_path=db_path,
                                  scalers=(input_scalers, output_scalers, static_scalers))

            preds, obs = self.predict_basin(ds_test)

            date_range = pd.date_range(start=self.cfg["start_date"]
                                       + pd.DateOffset(days=run_cfg["seq_length"] - 1),
                                       end=self.cfg["end_date"])
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()},
                              index=date_range)

            self.results[basin] = df

        return self.results

    def predict_basin(self, ds: LumpedBasin) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts a single basin.

        Parameters
        ----------
        ds : LumpedBasin
            Dataset for the basin to predict

        Returns
        -------
        preds : np.ndarray
            Array containing the (rescaled) network prediction for the entire data period
        obs : np.ndarray
            Array containing the observed discharge for the entire data period
        """
        preds, obs = self.model.predict(ds)
        preds = ds.output_scalers.rescale(preds)
        preds[preds < 0] = 0

        return preds, obs

    def get_nses(self) -> Dict:
        """Calculates the experiment's NSE for each calibration basin.

        Validation basins are ignored since they don't provide ground truth.

        Returns
        -------
        nses : Dict
            Dictionary mapping basin ids to their NSE

        Raises
        ------
        AttributeError
            If called before predicting
        """
        if len(self.results) == 0:
            raise AttributeError("No results to evaluate.")

        nses = {}
        for basin, df in self.results.items():
            # ignore validation basins that have no ground truth
            if not all(pd.isna(df['qobs'])):
                nses[basin] = nse(df['qsim'].values, df['qobs'].values)
        return nses
