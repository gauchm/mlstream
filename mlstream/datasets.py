from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .datautils import (load_discharge,
                        load_forcings_lumped,
                        load_static_attributes,
                        reshape_data)
from .scaling import InputScaler, OutputScaler, StaticAttributeScaler


class LumpedBasin(Dataset):
    """PyTorch data set to work with the raw text files for lumped (daily basin-aggregated)
    forcings and streamflow.

    Parameters
    ----------
    data_root : Path
        Path to the main directory of the data set
    basin : str
        Gauge-id of the basin
    forcing_vars : List
        Names of forcing variables to use
    dates : List
        Start and end date of the period.
    is_train : bool
        If True, discharge observations are normalized and invalid discharge samples are removed
    train_basins : List
        List of basins used in the training of the experiment this Dataset is part of. Needed to
        create the correct feature scalers (the ones that are calculated on these basins)
    seq_length : int
        Length of the input sequence
    with_attributes : bool, optional
        If True, loads and returns addtionaly attributes, by default False
    concat_static : bool, optional
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    db_path : str, optional
        Path to sqlite3 database file containing the catchment characteristics, by default None
    scalers : Tuple[InputScaler, OutputScaler, Dict[str, StaticAttributeScaler]], optional
        Scalers to normalize and resale input, output, and static variables. If not provided,
        the scalers will be initialized at runtime, which will result in poor performance if
        many datasets are created. Instead, it makes sense to re-use the scalers across datasets.
    """

    def __init__(self,
                 data_root: Path,
                 basin: str,
                 forcing_vars: List,
                 dates: List,
                 is_train: bool,
                 train_basins: List,
                 seq_length: int,
                 with_attributes: bool = False,
                 concat_static: bool = False,
                 db_path: str = None,
                 scalers: Tuple[InputScaler, OutputScaler,
                                Dict[str, StaticAttributeScaler]] = None):
        self.data_root = data_root
        self.basin = basin
        self.forcing_vars = forcing_vars
        self.seq_length = seq_length
        self.is_train = is_train
        self.train_basins = train_basins
        self.dates = dates
        self.with_attributes = with_attributes
        self.concat_static = concat_static
        self.db_path = db_path
        if scalers is not None:
            self.input_scalers, self.output_scalers, self.static_scalers = scalers
        else:
            self.input_scalers, self.output_scalers, self.static_scalers = None, None, {}
        if self.input_scalers is None:
            self.input_scalers = InputScaler(self.data_root, self.train_basins,
                                             self.dates[0], self.dates[1])
        if self.output_scalers is None:
            self.output_scalers = OutputScaler(self.data_root, self.train_basins,
                                               self.dates[0], self.dates[1])

        # placeholder to store std of discharge, used for rescaling losses during training
        self.q_std = None

        # placeholder to store start and end date of entire period (incl warmup)
        self.period_start = None
        self.period_end = None
        self.attribute_names = None

        self.x, self.y = self._load_data()

        if self.with_attributes:
            self.attributes = self._load_attributes()

        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.with_attributes:
            if self.concat_static:
                x = torch.cat([self.x[idx], self.attributes.repeat((self.seq_length, 1))], dim=-1)
                return x, self.y[idx]
            else:
                return self.x[idx], self.attributes, self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads input and output data from text files. """
        # we use (seq_len) time steps before start for warmup

        df = load_forcings_lumped(self.data_root, [self.basin])[self.basin]
        qobs = load_discharge(self.data_root, basins=[self.basin]).set_index('date')['qobs']
        if not self.is_train and len(qobs) == 0:
            tqdm.write(f"Treating {self.basin} as validation basin (no streamflow data found).")
            qobs = pd.Series(np.nan, index=df.index, name='qobs')

        df = df.loc[self.dates[0]:self.dates[1]]
        qobs = qobs.loc[self.dates[0]:self.dates[1]]
        if len(qobs) != len(df):
            print(f"Length of forcings {len(df)} and observations {len(qobs)} \
                  doesn't match for basin {self.basin}")
        df['qobs'] = qobs

        # store first and last date of the selected period
        self.period_start = df.index[0]
        self.period_end = df.index[-1]

        # use all meteorological variables as inputs
        x = np.array([df[var].values for var in self.forcing_vars]).T

        y = np.array([df['qobs'].values]).T

        # normalize data, reshape for LSTM training and remove invalid samples
        x = self.input_scalers.normalize(x)

        x, y = reshape_data(x, y, self.seq_length)

        if self.is_train:
            # Delete all samples where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                tqdm.write(f"Deleted {np.sum(np.isnan(y))} NaNs in basin {self.basin}.")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records with invalid discharge
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # store std of discharge before normalization
            self.q_std = np.std(y)

            y = self.output_scalers.normalize(y)

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _load_attributes(self) -> torch.Tensor:
        df = load_static_attributes(self.db_path, [self.basin], drop_lat_lon=True)

        # normalize data
        for feature in [f for f in df.columns if f[:7] != 'onehot_']:
            if feature not in self.static_scalers or self.static_scalers[feature] is None:
                self.static_scalers[feature] = \
                    StaticAttributeScaler(self.db_path, self.train_basins, feature)
            df[feature] = self.static_scalers[feature].normalize(df[feature])

        # store attribute names
        self.attribute_names = df.columns

        # store feature as PyTorch Tensor
        attributes = df.loc[df.index == self.basin].values
        return torch.from_numpy(attributes.astype(np.float32))


class LumpedH5(Dataset):
    """PyTorch data set to work with pre-packed hdf5 data base files.
    Should be used only in combination with the files processed from `create_h5_files` in the
    `utils` module.

    Parameters
    ----------
    h5_file : Path
        Path to hdf5 file, containing the bundled data
    basins : List
        List containing the basin ids
    db_path : str
        Path to sqlite3 database file, containing the catchment characteristics
    concat_static : bool
        If true, adds catchment characteristics at each time step to the meteorological forcing
        input data, by default False
    cache : bool, optional
        If True, loads the entire data into memory, by default False
    no_static : bool, optional
        If True, no catchment attributes are added to the inputs, by default False
    """

    def __init__(self,
                 h5_file: Path,
                 basins: List,
                 db_path: str,
                 concat_static: bool = False,
                 cache: bool = False,
                 no_static: bool = False):
        self.h5_file = h5_file
        self.basins = basins
        self.db_path = db_path
        self.concat_static = concat_static
        self.cache = cache
        self.no_static = no_static

        # Placeholder for catchment attributes stats
        self.df = None
        self.attribute_names = None

        # preload data if cached is true
        if self.cache:
            (self.x, self.y, self.sample_2_basin, self.q_stds) = self._preload_data()

        # load attributes into data frame
        self._load_attributes()

        # determine number of samples once
        if self.cache:
            self.num_samples = self.y.shape[0]
        else:
            with h5py.File(h5_file, 'r') as f:
                self.num_samples = f["target_data"].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if self.cache:
            x = self.x[idx]
            y = self.y[idx]
            basin = self.sample_2_basin[idx]
            q_std = self.q_stds[idx]

        else:
            with h5py.File(self.h5_file, 'r') as f:
                x = f["input_data"][idx]
                y = f["target_data"][idx]
                basin = f["sample_2_basin"][idx]
                basin = basin.decode("ascii")
                q_std = f["q_stds"][idx]

        if not self.no_static:
            # get attributes from data frame and create 2d array with copies
            attributes = self.df.loc[self.df.index == basin].values

            if self.concat_static:
                attributes = np.repeat(attributes, repeats=x.shape[0], axis=0)
                # combine meteorological obs with static attributes
                x = np.concatenate([x, attributes], axis=1).astype(np.float32)
            else:
                attributes = torch.from_numpy(attributes.astype(np.float32))

        # convert to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        q_std = torch.from_numpy(q_std)

        if self.no_static:
            return x, y, q_std
        else:
            if self.concat_static:
                return x, y, q_std
            else:
                return x, attributes, y, q_std

    def _preload_data(self):
        print("Preloading training data.")
        with h5py.File(self.h5_file, 'r') as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]
        return x, y, str_arr, q_stds

    def _get_basins(self):
        if self.cache:
            basins = list(set(self.sample_2_basin))
        else:
            with h5py.File(self.h5_file, 'r') as f:
                str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            basins = list(set(str_arr))
        return basins

    def _load_attributes(self):
        df = load_static_attributes(self.db_path, self.basins, drop_lat_lon=True)

        # normalize data
        self.attribute_scalers = {}
        for feature in [f for f in df.columns if f[:7] != 'onehot_']:
            self.attribute_scalers[feature] = \
                StaticAttributeScaler(self.db_path, self.basins, feature)
            df[feature] = self.attribute_scalers[feature].normalize(df[feature])

        # store attribute names
        self.attribute_names = df.columns
        self.df = df
