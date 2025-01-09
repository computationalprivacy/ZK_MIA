from preprocessing_helper_functions import *


if __name__ == '__main__':
    setting = 'Milano'
    region_name = 'Milano'
    # set shuffle=True to shuffle the roi and epoch indices (labels will be consistent across data objects)
    # this is done in order to securely share data with a third party
    shuffle = False
    n_seed = 2023
    num_processes = 4  # enter desired number of processes for parallelization

    # load raw milan data
    data_filepath = 'social-pulse-milano.geojson'
    with open(data_filepath, 'r') as geojson_file:
        milano_data = json.load(geojson_file)
    print(f'raw data has been loaded from {data_filepath}')

    features_milano = milano_data['features']
    user_data = []
    # create dictionaries from json data
    for feature in features_milano:
        properties = feature.get('properties', {})
        user_id = feature['user']
        timestamp = feature.get('timestamp', None)
        geom_point = feature.get('geomPoint.geom', {})
        if 'type' in geom_point and geom_point['type'] == 'Point':
            coordinates = geom_point.get('coordinates', [])
            if len(coordinates) >= 2:
                longitude, latitude = coordinates[:2]
                user_data.append({'user_id': user_id, 'timestamp': timestamp, 'latitude': latitude,
                                  'longitude': longitude})
    # initialize dataframe
    df_milano = pd.DataFrame(user_data)
    print(df_milano.head())

    # Due to sparsity, we will only keep the first week of data (up to 2013-11-07 21:59:59)
    milan_stop_time = 1383861599 # unix
    n_epochs = get_observation_period(df_milano, time_granularity=3600, stop_time_unix=milan_stop_time)

    # determine cutoffs containing 99% of the activity
    longitude_cutoffs, latitude_cutoffs, lon_extremes, lat_extremes = keep_longitude_latitude(df_milano)
    n_rois = get_observation_region(df_milano, latitude_cutoffs[0], latitude_cutoffs[1], longitude_cutoffs[0],
                                    longitude_cutoffs[1], grid_length=1.6)
    min_lat = latitude_cutoffs[0]
    max_lat = latitude_cutoffs[1]
    min_lon = longitude_cutoffs[0]
    max_lon = longitude_cutoffs[1]
    # apply time and space cutoffs
    df_milano = df_milano[df_milano['timestamp'] < milan_stop_time]
    df_milano = df_milano.loc[
        (df_milano['latitude'] < max_lat) & (df_milano['longitude'] < max_lon) & (df_milano['latitude'] > min_lat) & (
                    df_milano['longitude'] > min_lon)]
    # add a ROI column to the dataframe
    df_milano = add_roi_to_df(df_milano, 1.6, min_lon, max_lon, min_lat, max_lat)

    # re-label user-ids as integers
    milano_user_datetime_roi_ = df_milano[['user_id', 'roi', 'timestamp']]
    milano_user_datetime_roi = milano_user_datetime_roi_.copy(deep=True)
    for i, user in enumerate(set(milano_user_datetime_roi_['user_id'])):
        milano_user_datetime_roi.loc[milano_user_datetime_roi['user_id'] == user, 'user_id'] = i
    df_milano = milano_user_datetime_roi

    # we create a dataframe for the geographic positions (lon, let) of ROIs
    data_ = []
    for i in range(1, int(n_rois)+1):
        long, lat = lat_long_roi(i, latitude_cutoffs[0], longitude_cutoffs[0], latitude_cutoffs[1], longitude_cutoffs[1],
                                 grid_length=1.6, epsilon=1e-10)
        row = [i, long, lat]
        data_.append(row)
    milano_roi_lon_lat = pd.DataFrame(data_, columns=['roi', 'lon', 'lat'])

    print('preprocesssing to contain only data within the desired spatiotemporal setting from regular users')
    df_user, df_roi_map = preprocess_dfs(df_milano, milano_roi_lon_lat)
    rectangular_region = unpack_region(region_name)
    # Call the processing function with the provided arguments
    frac_in_setting, valid_users, rois_within_region, n_valid_visits_per_user, n_unique_rois_per_user = compute_statistics_in_setting(
        df_user, df_roi_map, num_processes)
    # filter and restructure the dataframes according to the spatiotemporal setting
    filtered_df_user, filtered_df_roi_map = filter_dataframes(df_user, df_roi_map, valid_users, rois_within_region)
    # add epoch column to the user level dataframe
    filtered_df_user = add_epoch_to_df_parallel(filtered_df_user, num_processes=num_processes)
    print(filtered_df_user.head())
    # print(filtered_df_roi_map.head())
    save_filename = save_results(region_name, filtered_df_user, filtered_df_roi_map, frac_in_setting,
                 valid_users, rois_within_region, n_valid_visits_per_user, n_unique_rois_per_user)

    # we will now create dictionary objects to save user data {user_id: location_trace, ... }
    # as well as a dictionary for a delaunay triangulation of the rois, to be used for MIAs later
    results_dictionary_pickle_name = save_filename
    save_dir = ''
    with open(results_dictionary_pickle_name, 'rb') as f:
        save_dic = pickle.load(f)
    df_user = save_dic['df_user']
    df_roi_map = save_dic['df_roi_map']
    print('df_user and df_roi_map loaded')

    # obtain desired dictionaries and shuffle maps

    data_dictionary, delaunay_nbrs_dict, user_id_mapping, roi_mapping, epoch_mapping = obtain_dictionaries(df_user, df_roi_map, n_seed, num_processes, shuffle=shuffle)
    output_directory = save_dir + f'data_dictionaries/'

    # create directory for saving aggregates if it doesn't already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # save dictionaries
    save_filename = f'data_dictionary_{setting}.pickle'
    output_file = os.path.join(output_directory, save_filename)
    with open(output_file,'wb') as f:
        pickle.dump(data_dictionary, f)
    print(f'data dictionary of user traces has been saved as {output_file}')
    save_filename = f'delaunay_nbrs_dict_{setting}.pickle'
    output_file = os.path.join(output_directory, save_filename)
    with open(output_file,'wb') as f:
        pickle.dump(delaunay_nbrs_dict, f)
    print(f'Delaunay dictionary of roi neighbourhoods has been saved as {output_file}')
    # save user, roi, epoch mappings for internal use if
    if shuffle:
        shuffled_maps_dict = {}
        shuffled_maps_dict['user'] = user_id_mapping
        shuffled_maps_dict['roi'] = roi_mapping
        shuffled_maps_dict['epoch'] = epoch_mapping
        print('user map:', user_id_mapping)
        print('roi map:', roi_mapping)
        print('epoch map:', epoch_mapping)
        save_filename = f'shuffled_mappings.pickle'
        output_file = os.path.join(output_directory, save_filename)
        with open(output_file, 'wb') as f:
            pickle.dump(shuffled_maps_dict, f)
        print(f'Shuffled maps for user, roi, and epoch data has been saved as {output_file}')

