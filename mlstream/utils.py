import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import h5py
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm

from .datasets import LumpedBasin
from .datautils import store_static_attributes


def create_h5_files(data_root: Path,
                    out_file: Path,
                    basins: List,
                    dates: List,
                    forcing_vars: List,
                    seq_length: int):
    """Creates H5 training set.

    Parameters
    ----------
    data_root : Path
        Path to the main directory of the data set
    out_file : Path
        Path of the location where the hdf5 file should be stored
    basins : List
        List containing the gauge ids
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    forcing_vars : List
        Names of forcing variables
    seq_length : int
        Length of the requested input sequences

    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset('input_data',
                                          shape=(0, seq_length, len(forcing_vars)),
                                          maxshape=(None, seq_length, len(forcing_vars)),
                                          chunks=True,
                                          dtype=np.float32,
                                          compression='gzip')
        target_data = out_f.create_dataset('target_data',
                                           shape=(0, 1),
                                           maxshape=(None, 1),
                                           chunks=True,
                                           dtype=np.float32,
                                           compression='gzip')

        q_stds = out_f.create_dataset('q_stds',
                                      shape=(0, 1),
                                      maxshape=(None, 1),
                                      dtype=np.float32,
                                      compression='gzip',
                                      chunks=True)

        sample_2_basin = out_f.create_dataset('sample_2_basin',
                                              shape=(0, ),
                                              maxshape=(None, ),
                                              dtype="S10",
                                              compression='gzip',
                                              chunks=True)

        scalers = None
        for basin in tqdm(basins, file=sys.stdout):
            dataset = LumpedBasin(data_root=data_root,
                                  basin=basin,
                                  forcing_vars=forcing_vars,
                                  is_train=True,
                                  train_basins=basins,
                                  seq_length=seq_length,
                                  dates=dates,
                                  scalers=scalers)
            # Reuse scalers across datasets to save computation time
            if scalers is None:
                scalers = dataset.input_scalers, dataset.output_scalers, dataset.static_scalers

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            # store input and output samples
            input_data.resize((total_samples, seq_length, len(forcing_vars)))
            target_data.resize((total_samples, 1))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            sample_2_basin.resize((total_samples, ))
            str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
            sample_2_basin[-num_samples:] = str_arr

            out_f.flush()


def store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Stores prediction results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.
    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"results_no_static_seed{run_cfg['seed']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"results_concat_static_seed{run_cfg['seed']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"results_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Successfully stored results at {file_name}")


def prepare_data(cfg: Dict, basins: List) -> Dict:
    """Pre-processes training data.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    basins : List
        List containing the gauge ids

    Returns
    -------
    Dict
        Dictionary containing the updated run config.
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = cfg["run_dir"] / "static_attributes.db"
    store_static_attributes(cfg["data_root"], db_path=cfg["db_path"],
                            attribute_names=cfg["static_attributes"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(data_root=cfg["data_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["start_date"], cfg["end_date"]],
                    forcing_vars=cfg["forcing_attributes"],
                    seq_length=cfg["seq_length"])

    return cfg


def setup_run(cfg: Dict) -> Dict:
    """Creates the folder structure for the experiment.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config

    Returns
    -------
    Dict
        Dictionary containing the updated run config
    """
    cfg["start_time"] = str(datetime.now())
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError('There is already a folder at {}'.format(cfg["run_dir"]))

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, Path):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            elif isinstance(val, np.ndarray):
                temp_cfg[key] = val.tolist()  # np.ndarrays are not serializable
            elif 'param_dist' in key:
                temp_dict = {}
                for k, v in val.items():
                    if isinstance(v, sp.stats._distn_infrastructure.rv_frozen):
                        temp_dict[k] = f"{v.dist.name}{v.args}, *kwds={v.kwds}"
                    else:
                        temp_dict[k] = str(v)
                temp_cfg[key] = str(temp_dict)
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def nse(qsim: np.ndarray, qobs: np.ndarray) -> float:
    """Calculates NSE, ignoring NANs in ``qobs``.

    .. math::
      \\text{NSE} =
      1 - \\frac{\\sum_{t=1}^T{(q_s^t - q_o^t)^2}}{\\sum_{t=1}^T{(q_o^t - \\bar{q}_o)^2}}

    Parameters
    ----------
    qsim : np.ndarray
        Predicted streamflow
    qobs : np.ndarray
        Ground truth streamflow

    Returns
    -------
    nse : float
        The prediction's NSE

    Raises
    ------
    ValueError
        If lenghts of qsim and qobs are not equal.
    """
    if len(qsim) != len(qobs):
        raise ValueError(f"Lenghts of qsim {len(qsim)} and qobs {len(qobs)} mismatch.")

    qsim = qsim[~np.isnan(qobs)]
    qobs = qobs[~np.isnan(qobs)]
    return 1 - (np.sum(np.square(qsim - qobs)) / np.sum(np.square(qobs - np.mean(qobs))))
