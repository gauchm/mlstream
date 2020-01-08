import re
from typing import Dict, List, Tuple
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import netCDF4 as nc
from numba import njit


def get_basin_list(data_root: Path, basin_type: str) -> List:
    """Returns the list of basin names.

    If basin_type is 'C' or 'V', the gauge_info.csv needs to contain a column
    'Cal_Val' that indicates the basin type.

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains a folder 'gauge_info'
        with the ``gauge_info.csv`` file
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

    gauge_info_file = data_root / 'gauge_info' / 'gauge_info.csv'
    gauge_info = pd.read_csv(gauge_info_file)
    if basin_type != '*':
        if 'Cal_Val' not in gauge_info.columns:
            raise RuntimeError('gauge_info.csv needs column "Cal_Val" to filter for basin types.')
        gauge_info = gauge_info[gauge_info['Cal_Val'] == basin_type]

    if 'Gauge_ID' not in gauge_info.columns:
        raise RuntimeError('gauge_info.csv has no column "Gauge_ID".')
    basins = gauge_info['Gauge_ID'].values

    return np.unique(basins).tolist()


def load_discharge(data_root: Path, basins: List = None, file_format: str = 'nc') -> pd.DataFrame:
    """Loads observed discharge for (calibration) gauging stations.

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains a directory 'discharge'
        with one or more nc-files.
    basins : List, optional
        List of basins for which to return data. If None (default), all basins are returned
    file_format : str, optional
        Format of the discharge files. Default, and currently only supported format is 'nc'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [date, basin, qobs], where 'qobs' contains the streamflow.
    """
    if file_format != 'nc':
        raise NotImplementedError(f"Discharge format {file_format} not supported.")

    discharge_dir = data_root / 'discharge'
    files = discharge_dir.glob('*.nc')

    data_streamflow = pd.DataFrame(columns=['date', 'basin', 'qobs'])
    found_basins = []
    for f in files:
        q_nc = nc.Dataset(f, 'r')
        file_basins = q_nc['station_id'][:]
        if basins is not None:
            # some basins might be in multiple NC-files. We only load them once.
            target_basins = [i for i, b in enumerate(file_basins)
                             if b in basins and b not in found_basins]
        else:
            target_basins = [i for i, b in enumerate(file_basins)
                             if b not in found_basins]
        if len(target_basins) > 0:
            time = nc.num2date(q_nc['time'][:], q_nc['time'].units, q_nc['time'].calendar)
            data = pd.DataFrame(q_nc['Q'][target_basins, :].T, index=time,
                                columns=file_basins[target_basins])
            data = data.unstack().reset_index().rename({'level_0': 'basin',
                                                        'level_1': 'date',
                                                        0: 'qobs'}, axis=1)
            found_basins += data['basin'].unique().tolist()
            data_streamflow = data_streamflow.append(data, ignore_index=True, sort=True)

        q_nc.close()

    return data_streamflow


def load_forcings_lumped(data_root: Path, basins: List = None, file_format: str = 'rvt') -> Dict:
    """Loads basin-lumped forcings.

    Parameters
    ----------
    data_root : Path
        Path to base data directory, which contains the directory 'forcings/lumped/',
        which contains one .rvt-file per basin.
    basins : List, optional
        List of basins for which to return data. Default (None) returns data for all basins.
    file_format : str, optional
        Format of the forcing files. Default, and currently only supported format is 'rvt'.

    Returns
    -------
    dict
        Dictionary of forcings (pd.DataFrame) per basin
    """
    if file_format != 'rvt':
        raise NotImplementedError(f"Forcing format {file_format} not supported.")

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
    f = data_root / 'gauge_info' / 'gauge_info.csv'
    gauge_info = pd.read_csv(f).rename({'Gauge_ID': 'basin'}, axis=1).set_index('basin')
    if attribute_names is not None:
        static_attributes = gauge_info.loc[:, attribute_names]
    else:
        static_attributes = gauge_info

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
        df = df.drop(['Lat_outlet', 'Lon_outlet'], axis=1)

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
