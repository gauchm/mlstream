import re
from typing import Dict, List, Tuple
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import netCDF4 as nc
from numba import njit


def get_basin_list(data_root: Path, basin_type: str) -> List:
    """Returns the list of basin names

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains a folder 'gauge_info'
        with one or more csv-files.
    basin_type : str
        'C' to return calibration stations only,
        'V' to return validation stations only,
        '*' to return all stations

    Returns
    -------
    list
        List of basin name strings
    """
    if basin_type not in ['*', 'C', 'V']:
        raise ValueError('Illegal basin type')

    gauge_info_dir = data_root / 'gauge_info'
    files = gauge_info_dir.glob('*.csv')
    basins = np.array([], dtype=str)
    for f in files:
        gauge_info = pd.read_csv(f)
        if basin_type != '*':
            gauge_info = gauge_info[gauge_info['Calibration/Validation'] == basin_type]
        basins = np.concatenate([basins, gauge_info['ID'].values])

    return np.unique(basins).tolist()


def load_discharge(data_root: Path, basins: List = None) -> pd.DataFrame:
    """Loads observed discharge for (calibration) gauging stations.

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains a directory 'discharge'
        with one or more nc-files.
    basins : List, optional
        List of basins for which to return data. If None (default), all basins are returned

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [date, basin, qobs], where 'qobs' contains the streamflow.
    """
    discharge_dir = data_root / 'discharge'
    files = discharge_dir.glob('*.nc')

    data_streamflow = None
    for f in files:
        q_nc = nc.Dataset(f, 'r')
        f_basins = q_nc['station_id'][:]
        target_basins = [i for i, b in enumerate(f_basins) if b in basins] if basins is not None \
            else range(len(f_basins))
        if len(target_basins) > 0:
            time = nc.num2date(q_nc['time'][:], q_nc['time'].units, q_nc['time'].calendar)
            data = pd.DataFrame(q_nc['Q'][target_basins, :].T, index=time,
                                columns=f_basins[target_basins])
            if data_streamflow is None:
                data_streamflow = data
            else:
                # some basins might be in multiple NC-files. We only load them once.
                data = data[[s for s in f_basins if s not in data_streamflow.columns]]
                data_streamflow = data_streamflow.join(data)
        q_nc.close()

    data_streamflow = data_streamflow.loc['2000-01-01':'2016-12-31'].unstack().reset_index()\
        .rename({'level_0': 'basin', 'level_1': 'date', 0: 'qobs'}, axis=1)

    if basins is not None:
        data_streamflow = data_streamflow[data_streamflow['basin'].isin(basins)]\
            .reset_index(drop=True)
    return data_streamflow


def load_forcings_lumped(data_root: Path, basins: List = None) -> Dict:
    """Loads basin-lumped forcings.

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains the directory 'forcings/lumped/',
        which contains one .rvt-file per basin.
    basins : List, optional
        List of basins for which to return data. Default (None) returns data for all basins.

    Returns
    -------
    dict
        Dictionary of forcings (pd.DataFrame) per basin
    """
    lumped_dir = data_root / 'forcings' / 'lumped'
    basin_files = lumped_dir.glob('*.rvt')

    basin_forcings = {}
    for f in basin_files:
        basin = f.name.split('_')[-1][:-4]
        if basins is not None and basin not in basins:
            continue

        with open(f) as fp:
            next(fp)
            start_date = next(fp)[:10]
            columns = re.split(r',\s+', next(fp).replace('\n', ''))[1:]
        data = pd.read_csv(f, sep=r',\s*', skiprows=4, skipfooter=1, names=columns, dtype=float,
                           header=None, usecols=range(len(columns)), engine='python')

        data.index = pd.date_range(start_date, periods=len(data), freq='D')
        basin_forcings[basin] = data

    return basin_forcings


def store_static_attributes(data_root: Path, db_path: Path = None, attribute_names: List = None):
    """Loads catchment characteristics from text file and stores them in a sqlite3 table

    Parameters
    ----------
    data_root : Path
        Path to the main directory of the data set
    db_path : Path, optional
        Path to where the database file should be saved. If None, stores the database in
        data_root/static_attributes.db. Default: None
    attribute_names : List, optional
        List of attribute names to use. Default: use all attributes.

    Raises
    ------
    RuntimeError
        If attributes folder can not be found.
    """
    static_attributes = pd.DataFrame()
    gauge_info_dir = data_root / 'gauge_info'
    for f in gauge_info_dir.glob('*.csv'):
        gauge_info = pd.read_csv(f).rename({'ID': 'basin'}, axis=1).set_index('basin')
        new_basins = [b for b in gauge_info.index.values if b not in static_attributes.index.values]
        if attribute_names is None:
            attribute_names = gauge_info.columns
        static_attributes = static_attributes.append(gauge_info.loc[new_basins, attribute_names])

    if db_path is None:
        db_path = data_root / 'static_attributes.db'

    with sqlite3.connect(str(db_path)) as conn:
        # insert into databse
        static_attributes.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_static_attributes(db_path: Path,
                           basins: List,
                           drop_lat_lon: bool = True,
                           keep_features: List = None) -> pd.DataFrame:
    """Loads attributes from database file into DataFrame and one-hot-encodes non-numerical features.

    Parameters
    ----------
    db_path : Path
        Path to sqlite3 database file
    basins : List
        List containing the basin id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is basin id.
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='basin')

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['Lat', 'Lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
        df = df.drop(drop_names, axis=1)

    # one-hot-encoding
    non_numeric_features = [f for f in df.columns if df[f].dtype == object]
    df = pd.get_dummies(df, columns=non_numeric_features,
                        prefix=[f"onehot_{f}" for f in non_numeric_features])

    # drop rows of basins not contained in data set
    df = df.loc[basins]

    return df


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new : np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new : np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new