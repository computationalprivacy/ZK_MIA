import pandas as pd
import numpy as np
import os
import json
import datetime as dt
import argparse
import scipy.spatial as sp
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
import multiprocessing as mp
import pickle
from random import seed, sample
from collections import defaultdict
from datetime import datetime
import time
from math import radians, cos, sin, asin, sqrt, floor

def get_observation_period(df, time_granularity=3600, stop_time_unix=1383861599):
    """
    This function reports the observation period of visits captured within the raw dataframe
    df: DataFrame of all users' geotagged visits. Assumes there is a "timestamp" column providing times in unix
    time_granularity: Duration of an epoch, measured in seconds. Default is 3600 seconds (1 hour)
    stop_time_unix: The Unix timestamp until which data should be considered.
    """
    # Assert that the 'timestamp' column exists
    assert 'timestamp' in df.columns, "DataFrame does not contain a 'timestamp' column."

    # Assert that the 'timestamp' column contains Unix time (integers or floats)
    assert df['timestamp'].dtype in ['int64',
                                     'float64'], "'timestamp' column must be in Unix format"

    # Get the minimum and maximum timestamps from the dataset
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()

    # Convert timestamps to datetime strings for display
    min_time_dt = datetime.utcfromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S')
    max_time_dt = datetime.utcfromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S')

    # Print dataset observation period
    print(f'The full dataset has observations from {min_time_dt} to {max_time_dt}')
    print(f'We will keep data from {min_time_dt} up until {datetime.utcfromtimestamp(stop_time_unix).strftime("%Y-%m-%d %H:%M:%S")}.')

    # Calculate the number of epochs between min_time and stop_time_unix
    n_epochs = int((stop_time_unix - min_time) / time_granularity) + 1
    print(f'There will be {n_epochs} epochs of length {time_granularity / 3600} hours from {min_time_dt} to {datetime.utcfromtimestamp(stop_time_unix).strftime("%Y-%m-%d %H:%M:%S")}')
    return n_epochs

def compute_epoch(timestamp, min_time, time_granularity):
    '''
    Returns the corresponding epoch value for the timestamp with respect to min_time and time_granularity
    timestamp, min_time, and time_granularity are in unix
    time_granularity is in seconds. Default is to use 3600 = one hour per epoch
    '''
    rounded_t = time_granularity * np.floor(timestamp / time_granularity)
    epoch = int((rounded_t - min_time) / time_granularity)
    return epoch

def keep_longitude_latitude(df):
    '''
    Returns the longitude and latitude cutoffs to keep 99% of the data from the dataframe
    '''
    assert 'latitude' in df.columns, "DataFrame does not contain a 'latitude' column."
    assert 'longitude' in df.columns, "DataFrame does not contain a 'longitude' column."
    max_lat = df['latitude'].max()
    min_lat = df['latitude'].min()
    max_lon = df['longitude'].max()
    min_lon = df['longitude'].min()
    lon_extremes = (min_lon, max_lon)
    lat_extremes = (min_lat, max_lat)
    print(f'The dataset has observations within [{min_lat} , {max_lat}] x  [{min_lon} , {max_lon}]')

    # Get the longitude range where 99% of points lie
    percentile_1 = np.percentile(df['longitude'], 0.5)
    percentile_2 = np.percentile(df['longitude'], 99.5)
    longitude_cutoff_1 = percentile_1
    longitude_cutoff_2 = percentile_2
    longitude_cutoffs = (longitude_cutoff_1, longitude_cutoff_2)

    # Get the latitude range where 99% of points lie
    percentile_1 = np.percentile(df['latitude'], 0.5)
    percentile_2 = np.percentile(df['latitude'], 99.5)
    latitude_cutoff_1 = percentile_1
    latitude_cutoff_2 = percentile_2
    latitude_cutoffs = (latitude_cutoff_1, latitude_cutoff_2)

    print(f'99% Cutoffs for Longitudes: {longitude_cutoffs}')
    print(f'99% Cutoffs for Latitudes: {latitude_cutoffs}')

    return longitude_cutoffs, latitude_cutoffs, lon_extremes, lat_extremes


def get_observation_region(df, min_lat, max_lat, min_lon, max_lon, grid_length=None):
    '''
    grid_length is in km. Default is 1.6km for Milano dataset
    '''
    print(f'We consider observations within [{min_lat} , {max_lat}] x  [{min_lon} , {max_lon}]')
    if grid_length is not None:
        # overestimates the region as a rectangle defined by the previous 4 extreme coordinates
        region_area = compute_area(min_lat, max_lat, min_lon, max_lon)
        n_rois = region_area // (grid_length ** 2) + 1
        print(f'There should be {n_rois} ROIs (squares of area {grid_length ** 2} km^2)')
        return n_rois


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def compute_area(minlat, maxlat, minlon, maxlon):
    d_lat = haversine(minlon, minlat, minlon, maxlat)
    d_lon = haversine(minlon, minlat, maxlon, minlat)
    return d_lon * d_lat


def compute_roi(longitude, latitude, grid_length, minlon, maxlon, minlat, maxlat):
    # Calculate the number of grid cells in longitude and latitude directions
    grid_cells_lon = haversine(minlon, minlat, maxlon, minlat) // grid_length
    grid_cells_lat = haversine(minlon, minlat, minlon, maxlat) // grid_length
    # Calculate the indices of the grid cell in longitude and latitude directions
    lon_index = int(haversine(minlon, minlat, longitude, minlat) // grid_length)
    lat_index = int(haversine(minlon, minlat, minlon, latitude) // grid_length)
    # Calculate the ROI by mapping the grid cell indices
    roi = (grid_cells_lon * lat_index) + lon_index
    return int(roi) + 1


def add_roi_to_df(df, grid_length, min_lon, max_lon, min_lat, max_lat):
    df['roi'] = df.apply(
        lambda row: compute_roi(row['longitude'], row['latitude'], grid_length, min_lon, max_lon, min_lat, max_lat),
        axis=1)
    return df


def lat_long_roi(roi_idx, minlat, minlon, maxlat, maxlon, grid_length, epsilon):
    grid_cells_lon = haversine(minlon, minlat, maxlon, minlat) // grid_length
    grid_cells_lat = haversine(minlon, minlat, minlon, maxlat) // grid_length

    lat_index = roi_idx // grid_cells_lon
    lon_index = roi_idx % grid_cells_lat

    lon_start = minlon + (lon_index * grid_length / haversine(minlon, minlat, minlon + 1, minlat))
    lat_start = minlat + (lat_index * grid_length / haversine(minlon, minlat, minlon, minlat + 1))

    lon_centroid = lon_start + (0.5 * grid_length / haversine(minlon, minlat, minlon + 1, minlat))
    lat_centroid = lat_start + (0.5 * grid_length / haversine(minlon, minlat, minlon, minlat + 1))
    if lon_centroid > maxlon:
        lon_centroid = maxlon - epsilon
    if lon_centroid < minlon:
        lon_centroid = minlon + epsilon
    if lat_centroid > maxlat:
        lat_centroid = maxlat - epsilon
    if lat_centroid < minlat:
        lat_centroid = minlat + epsilon

    return lon_centroid, lat_centroid

def preprocess_dfs(df_user, df_roi_map):
    assert 'user_id' in df_user.columns, "df_user does not contain a 'user_id' column."
    df_user.drop(index=df_user.index[0], axis=0, inplace=True)
    # df_user['timestamp'] = df_user['timestamp'].apply(
    #     lambda timer: int(time.mktime(dt.datetime.strptime(timer, '%Y-%m-%d %H:%M:%S').timetuple())))
    df_roi_map['lon'] = df_roi_map['lon'].apply(lambda data: float(data))
    df_roi_map['lat'] = df_roi_map['lat'].apply(lambda data: float(data))
    df_roi_map['roi'] = df_roi_map['roi'].apply(lambda data: int(data)-1)
    df_user['roi'] = df_user['roi'].apply(lambda data: int(data) - 1)
    df_user['user_id'] = df_user['user_id'].apply(lambda data: int(data) + 1)
    check_dfs_format(df_user, df_roi_map)
    return df_user, df_roi_map

def unix_to_str(timestamp):
    datetime_object = dt.datetime.fromtimestamp(timestamp)
    return datetime_object.strftime('%Y-%m-%d %H:%M:%S')

def check_dfs_format(df_user, df_roi_map):
    '''
    This function checks that df_user and df_roi_map obey the right format
    '''
    user_column_names = set(df_user.columns)
    assert user_column_names == {'user_id', 'timestamp', 'roi'}, 'df_user must have columns: user_id, time, roi.'
    roi_map_column_names = set(df_roi_map.columns)
    assert roi_map_column_names == {'roi', 'lon', 'lat'}, 'df_roi_map must have columns: roi, lon, lat.'
    # Check if all values in the time column are in Unix format
    assert all(isinstance(value, int) for value in df_user['timestamp']), 'time column of df_user must be in UNIX (int)'
    df_user_roi_set = set(df_user['roi'])
    df_roi_map_roi_set = set(df_roi_map['roi'])
    assert df_user_roi_set.issubset(
        df_roi_map_roi_set), 'df_user has observations of rois that are not accounted for in df_roi_map'
    # Check if all values in the lon column are floats
    assert all(
        isinstance(value, float) for value in df_roi_map['lon']), 'lon column of df_roi_map must be in UNIX (int)'
    # Check if all values in the lat column are floats
    assert all(
        isinstance(value, float) for value in df_roi_map['lat']), 'lat column of df_roi_map must be in UNIX (int)'
    assert df_roi_map['roi'].min() == 0, 'rois must be 0-indexed'
    assert df_user['user_id'].min() == 1, 'user_ids must be 0-indexed'
    print('both inputted dataframes passed all checks')

def unpack_observation_period(period_name):
    if period_name == 'November 2013':
        min_time = 1383284733 # 2013-10-31 22:45:32
        max_time = 1383861599 # 2013-11-07 21:59:59
    print(f'Observation period has been set to {period_name}: [{unix_to_str(min_time)},{unix_to_str(max_time)}]')
    return (min_time, max_time)

def unpack_region(region_name):
    if region_name == 'Milano':
        min_lon = 9.07954122975
        max_lon = 9.27899377165
        min_lat = 45.394252415050005
        max_lat = 45.541507186800004
    print(f'Region has been set to {region_name}')
    return (min_lon, max_lon, min_lat, max_lat)


def process_user(user_df_id, rois_within_region):
    '''
    Parallelization helper function for compute_statistics_in_setting
    Inputs:
    user_df_id: (user_id, user_df), user_df is the part of df_user corresponding only to user_id
    rois_within_region: list of valid roi ids (list of int)
    Outputs:
    If user is deemed valid: (user_id, visits_in_region, n_rois_visited)
    '''
    user_id, user_df = user_df_id
    total_count = 0
    visits_in_region = 0
    rois_visited = []
    for i, row in user_df.iterrows():
        roi = user_df.loc[i, 'roi']
        total_count += 1
        if roi in rois_within_region:
            visits_in_region += 1
            if roi not in rois_visited:
                rois_visited.append(roi)
    if visits_in_region >= 0.5*total_count:
        return (user_id, visits_in_region, len(rois_visited))
    else:
        return None


def compute_statistics_in_setting(df_user, df_map, num_processes):
    '''
    Inputs:
    df_user: df for user level data
    df_map: df that maps each roi to lon, lat coordinates
    time_interval: tuple of two ordered unix timestamps defining observation period
    rectangular_region: tuple of four floats min_lon, max_lon, min_lat, max_lat)
    num_processes: int (for parallelization)
    Outputs:
    frac_in_setting: fraction of valid users over total users in df_user
    valid_users: list of user_ids of valid users, i.e. the majority of their trace during time_interval is within rectangular_region)
    rois_within_region: roi ids corresponding to rois that are within rectangular_region
    n_valid_visits_per_user: list of number of valid observations per valid user within the spatiotemporal setting
    n_unique_rois_per_user: list of number of uniquely visited rois per valid user within the spatiotemporal setting
    '''
    # separate the df by user
    user_dfs_by_id = df_user.groupby('user_id')
    # Get the valid roi ids
    rois_within_region = df_map['roi'].to_list()
    with mp.Pool(num_processes) as pool:
        # result is either a tuple (user_id, visits_in_region) or None (if the user is not deemed valid)
        results = pool.starmap(process_user, [(user_df_id, rois_within_region) for user_df_id in user_dfs_by_id])
    pool.close()
    pool.join()
    n_valid_visits_per_user = []
    n_unique_rois_per_user = []
    valid_users = []
    for result in results:
        if result is not None:
            valid_users.append(result[0])
            n_valid_visits_per_user.append(result[1])
            n_unique_rois_per_user.append(result[2])
    n_valid_users = len(valid_users)
    frac_in_setting = n_valid_users / len(user_dfs_by_id)
    return frac_in_setting, valid_users, rois_within_region, n_valid_visits_per_user, n_unique_rois_per_user


def filter_dataframes(df_user, df_roi_map, valid_users, rois_within_region):
    # # Keep only the valid users
    mask_valid_user = df_user['user_id'].isin(valid_users)
    # # Keep only visits reported within the appropriate region
    mask_valid_roi = df_user['roi'].isin(rois_within_region)
    # Keep only visits reported during the observation period [min_time, max_time]
    combined_mask = mask_valid_user & mask_valid_roi
    # Filter df_user
    filtered_df_user = df_user[combined_mask]
    mask_valid_roi = df_roi_map['roi'].isin(rois_within_region)
    # Filter df_roi_map
    filtered_df_roi_map = df_roi_map[mask_valid_roi]
    # Now, we restructure rois to be ordered from 0 to n_rois-1 within both dfs
    filtered_df_user = filtered_df_user.reset_index(drop=True)
    # Create a mapping dictionary for roi values
    roi_mapping = {old_roi: new_roi for new_roi, old_roi in enumerate(filtered_df_roi_map['roi'].unique())}
    # Apply mapping to roi column in df_user
    filtered_df_user['roi'] = filtered_df_user['roi'].map(roi_mapping)
    # Apply mapping to roi column in df_roi_map
    filtered_df_roi_map['roi'] = filtered_df_roi_map['roi'].map(roi_mapping)
    return filtered_df_user, filtered_df_roi_map


def compute_epoch(t, min_time, time_granularity):
    '''
    Computes the epoch (0-indexed) of a unix time t given time_granularity (in seconds) and the unix representation of the minimum cutoff time
    '''
    rounded_t = time_granularity * floor(t / time_granularity)
    epoch = int((rounded_t - min_time) / time_granularity)
    return epoch


def process_chunk_epoch(chunk, min_time, time_granularity):
    '''
    Parallelization helper function for converting unix time values to epoch (0-indexed)
    '''
    chunk['epoch'] = chunk['timestamp'].apply(compute_epoch, args=(min_time, time_granularity))
    return chunk


def add_epoch_to_df_parallel(df, time_granularity=3600, min_time=None, num_processes=1):
    '''
    Adds a corresponding epoch column to df. Default is to use one hour bins for epochs
    '''
    if min_time is None:
        min_time = df['timestamp'].min()
    pool = mp.Pool(num_processes)
    results = pool.starmap(
        process_chunk_epoch,
        [(chunk, min_time, time_granularity) for chunk in np.array_split(df, num_processes)]
    )
    # Combine the processed chunks back into a single dataframe
    result_df = pd.concat(results)
    pool.close()
    pool.join()
    return result_df


def check_args(args):
    assert args.region_name in ['Milano'], \
        f'{args.region_name} is an unsupported region name.'
    assert args.period_name in ['November 2013'], \
        f'{args.period_name} is an unsupported observation period name.'


def save_results(region_name, filtered_df_user, filtered_df_roi_map, frac_in_setting, valid_users,
                 rois_within_region, n_valid_visits_per_user, n_unique_rois_per_user):
    print(
        f'Percent of users whose majority of visits are within {region_name}: {round(frac_in_setting * 100, 2)}%')
    print(f'Total number of valid users: {len(valid_users)}')
    print(f'Number of rois: {len(rois_within_region)}')
    save_dic = {}
    save_dic['region'] = region_name
    save_dic['df_user'] = filtered_df_user
    save_dic['df_roi_map'] = filtered_df_roi_map
    save_dic['frac_in_setting'] = frac_in_setting
    save_dic['valid_users'] = valid_users
    save_dic['rois_within_region'] = rois_within_region
    save_dic['n_valid_visits_per_user'] = n_valid_visits_per_user
    save_dic['n_unique_rois_per_user'] = n_unique_rois_per_user
    save_filename = f'preliminary_statistics_{region_name}.pickle'
    with open(save_filename, 'wb') as f:
        pickle.dump(save_dic, f)
    print(f'dictionary of preliminary statistics has been saved as {save_filename}')
    return save_filename


def obtain_dimensions(df_user):
    '''
    Obtain the actual number of rois and epochs that report visits in df_user
    '''
    n_rois = df_user['roi'].nunique()
    print('number of rois that were visited at least once:', n_rois)
    n_epochs = df_user['epoch'].nunique()
    print('number of epochs that were visited at least once:', n_epochs)
    return n_rois, n_epochs


def create_user_trace(user_id, user_df, n_rois, n_epochs):
    user_trace = lil_matrix((n_rois, n_epochs), dtype=np.int8)
    # Iterate over the user dataframe and set the corresponding matrix entries to 1
    for _, row in user_df.iterrows():
        roi = row['roi']
        epoch = row['epoch']
        user_trace[roi, epoch] = 1
    return user_id, user_trace.tocsr()


def apply_delaunay_clustering(df_roi_map):
    """
    This function reads lon and lat information from a df and returns
    a dictionary linking antennas (rois) to neighbouring antennas (rois).

    Inputs:
        - df_roi_map: pandas df containing roi/antenna ids, lon, and lat
    Outputs:
        - dict with key = roi id and with value = set of neighbouring roi ids 
    """
    tower_dict = dict(df_roi_map.groupby(['lon', 'lat']).indices)
    points, ant_in_tower = zip(*tower_dict.items())
    ant_in_tower = [np.int16(i) for i in ant_in_tower]
    points = np.array(points)
    tri = sp.Delaunay(points)
    ant_neighbour_ant = defaultdict(set)
    for i in range(len(points)):
        t_neighbours = find_neighbors(i, tri)
        a = set()
        for t in t_neighbours:
            ants = ant_in_tower[t]
            a = a.union(set(ants))
            for ant in ant_in_tower[i]:
                ant_neighbour_ant[ant] = a
    return dict(ant_neighbour_ant)


def find_neighbors(pindex, triang):
    """This function uses the built in scipy.spatial.Delaunay objects features
    to fetch the neighbouring nodes of a particular point.

    Inputs:
        - pindex: int, index of point to find neighbours for
        - triang: scipy.spatial.Delaunay object

    Outputs:
        - ndarray of inidces of points which neighbour pindex
    """
    a = triang.vertex_neighbor_vertices[1]
    b = triang.vertex_neighbor_vertices[0][pindex]
    c = triang.vertex_neighbor_vertices[0][pindex + 1]
    return a[b:c]


def obtain_dictionaries(df_user, df_roi_map, n_seed, num_processes, shuffle = False):
    '''
    Produces data_dictionary = {user_id: trace as csr_matrix, ... }
    and delaunay_nbrs_dict = {roi: set(nbrs(roi)), ... }
    where roi ids and epoch ids have been reduced to {0, ..., n_rois-1} and {0, ..., n_epochs-1} respectively and shuffled
    '''
    n_rois, n_epochs = obtain_dimensions(df_user)
    # Create a mapping for the 'roi' column to shift to {0, ..., n_rois-1}
    roi_mapping = {roi: new_roi for new_roi, roi in enumerate(df_user['roi'].unique())}
    # Create a mapping for the 'epoch' column to shift to {0, ..., n_epochs-1}
    epoch_mapping = {epoch: new_epoch for new_epoch, epoch in enumerate(df_user['epoch'].unique())}
    if shuffle:
        # set seed for reproducibility
        seed(n_seed)
        # shuffle the rois and epochs by reordering the values
        roi_mapping = dict(zip(roi_mapping.keys(), sample(list(roi_mapping.values()), len(roi_mapping))))
        # print('max in roi map:', max(roi_mapping_shuffled))
        epoch_mapping = dict(zip(epoch_mapping.keys(), sample(list(epoch_mapping.values()), len(epoch_mapping))))

    # Update df_user accordingly
    df_user['roi'] = df_user['roi'].map(roi_mapping)
    # print('max in df_user:', max(df_user['roi'].values))
    df_user['epoch'] = df_user['epoch'].map(epoch_mapping)
    # Update df_roi_map accordingly
    df_roi_map['roi'] = df_roi_map['roi'].map(roi_mapping)
    # deletes the antennas without visits (nan)
    df_roi_map = df_roi_map.dropna(subset=['roi'])
    # df_roi_map['roi'] = df_roi_map['roi'].astype(int)
    # Split df_user up by user_id
    df_user_grouped = df_user.groupby('user_id')
    # this block of code creates a data dictionary {user_id: trace as csr_matrix, ... }
    print('creating user traces')
    with mp.Pool(num_processes) as pool:
        # result is tuple (user_id, user_trace) 
        results = pool.starmap(create_user_trace, [(user_id, user_df, n_rois, n_epochs) for user_id, user_df in df_user_grouped])
    data_dictionary = {}
    user_id_mapping = {}  # New mapping from original user_id to contiguous index in 1, ..., n_users
    index_counter = 1  # Counter for contiguous index
    print('saving user traces to data dictionary')
    for result in tqdm(results):
        user_id = result[0]
        user_trace = result[1]
        data_dictionary[index_counter] = user_trace
        user_id_mapping[user_id] = index_counter
        index_counter += 1
    # this block of code creates a delaunay neighbourhood dictionary {roi: set(nbrs(roi)), ... }
    print('creating delaunay triangulation')
    delaunay_nbrs_dict = apply_delaunay_clustering(df_roi_map)
    return data_dictionary, delaunay_nbrs_dict, user_id_mapping, roi_mapping, epoch_mapping


