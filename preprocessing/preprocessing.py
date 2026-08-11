# load packages
import os
import glob
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import warnings

import pyeto

var_names = ['RDC','WT','SC']
MET_vars = ['precip','temp']


def get_dirs():
    """Define and create all input/output directory paths."""
    base_dir = Path.cwd()

    inputs_dir  = base_dir / "INPUTS"
    outputs_dir = base_dir / "OUTPUTS"

    dirs = {
        'base_dir':                          base_dir,
        'inputs_dir':                        inputs_dir,
        'outputs_dir':                       outputs_dir,
        'gagesii_dir':                       inputs_dir / "GAGESII",
        'met_raw_dir':                       inputs_dir / "MET_RAW",
        'nlcd_raw_dir':                      inputs_dir / "NLCD_RAW",
        'rdc_wt_sc_raw_dir':                 inputs_dir / "RDC_WT_SC_RAW",
        'upper_colorado_river_boundary_dir': inputs_dir / "Upper_Colorado_River_Basin_Boundary",
        'wt_lstm_data_dir':                  inputs_dir / "WT_LSTM_data",
        'reservoir_dir':                     inputs_dir / "RESERVOIR_ALL" / "RESERVOIR_RAW",
        'met_data_dir':                      outputs_dir / "MET_data",
        'nlcd_data_dir':                     outputs_dir / "NLCD_data",
        'rdc_wt_sc_data_dir':               outputs_dir / "RDC_WT_SC_data",
        'spei_data_dir':                     outputs_dir / "SPEI_data",
        'reservoir_data_dir':                outputs_dir / "RESERVOIR_data",
    }

    # Create output directories if they don't exist
    for key in ['met_data_dir', 'nlcd_data_dir', 'rdc_wt_sc_data_dir',
                'spei_data_dir', 'reservoir_data_dir']:
        dirs[key].mkdir(parents=True, exist_ok=True)

    return dirs


def load_all_inputs():
    """
    Load all raw input data and return directories + dataframes.
    This is the single entry point called by the preprocessing notebook.
    """
    dirs = get_dirs()

    # File paths
    raw_met_data_path    = dirs['met_raw_dir']      / "HUC14_OREGONSTATE-PRISM-AN81m_tmean_ppt_1971-01-01_2024-09-01.csv"
    gagesii_path         = dirs['gagesii_dir']      / "gagesII_metadata.csv"
    gagesii_traits_path  = dirs['gagesii_dir']      / "processed_dataset.csv"
    augmentedWT_path     = dirs['wt_lstm_data_dir'] / "full_274attr_lstm_outputs_all_huc_14_sites.csv"
    RMSE_WT_path         = dirs['wt_lstm_data_dir'] / "full_274attr_lstm_RMSEs_all_huc_14_sites.csv"

    # Load data
    raw_met_data     = pd.read_csv(raw_met_data_path)
    gagesii          = pd.read_csv(gagesii_path).set_index('siteid')
    gage_loc         = gagesii[['LAT_GAGE']]
    augmentedWT_data = pd.read_csv(augmentedWT_path, index_col=0)
    RMSE_WT_data     = pd.read_csv(RMSE_WT_path,     index_col=0)

    return dirs, raw_met_data, gagesii, gage_loc, augmentedWT_data, RMSE_WT_data


def load_usgs_basin3d_data (huc: int, prefix, start_date: str, end_date: str, RDC_WT_SC_data_dir=None, filename_suffix="_DAY.h5"):
    """
    Load data from USGS CONUS DOWNLOADS stored as a BASIN-3D HDF5 file into a pandas data frame

    Parameters
    :param huc: The HUC region that the data file is specified for (assumes that the usgs data files are separated by HUC)
    :prefix: Any additional prefix?
    :start_date: Start date of the query
    :end_date: End date of the query
    :filename_suffix: Default .h5, but can override with other extensions

    Returns data and metadata separately
    """

    filename = "USGS-" + huc +"_"+prefix+"_" + start_date + "_" + end_date + filename_suffix
    data = pd.read_hdf(RDC_WT_SC_data_dir / filename, key='data')
    metadata = pd.read_hdf(RDC_WT_SC_data_dir / filename, key='metadata')
    print("Returning the data and metadata extracted by the BASIN3D data loader.")
    print('-' * 42)
    return data, metadata


def sep_min_mean_max(df, var):
    """
    Separate DataFrame columns into MIN, MEAN, and MAX 

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    var (str): Variable name to filter columns.

    Returns:
    tuple: Tuple containing the MIN, MEAN, and MAX DataFrames.
    """
    min_df = df.filter(regex=f'{var}__MIN$')
    mean_df = df.filter(regex=f'{var}__MEAN$')
    max_df = df.filter(regex=f'{var}__MAX$')

    min_df.columns = min_df.columns.str.replace(f'__{var}__MIN', '')
    mean_df.columns = mean_df.columns.str.replace(f'__{var}__MEAN', '')
    max_df.columns = max_df.columns.str.replace(f'__{var}__MAX', '')

    min_df.columns = [col[5:] for col in min_df.columns]
    mean_df.columns = [col[5:] for col in mean_df.columns]
    max_df.columns = [col[5:] for col in max_df.columns]

    return min_df, mean_df, max_df


def split_datetime(df):
    """
    For a given DataFrame with a DatetimeIndex, return a DataFrame with a MultiIndex ['year', 'month', 'day'].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
    
    Returns:
    pd.DataFrame: DataFrame with a MultiIndex ['year', 'month', 'day'].
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    
    # Create a MultiIndex from the index
    multi_index = pd.MultiIndex.from_arrays(
        [df.index.year, df.index.month, df.index.day],
        names=['year', 'month', 'day']
    )
    
    # Set the new MultiIndex
    df.index = multi_index
    return df


def convert_to_water_years(dataset):
    """
    Convert a pandas DataFrame with a MultiIndex ['year', 'month', 'day'] or ['year', 'month']
    to a DataFrame with a MultiIndex ['wyear', 'month', 'day'] or ['wyear', 'month'].

    Parameters:
    dataset (pd.DataFrame): Input DataFrame with a MultiIndex ['year', 'month', 'day'] or ['year', 'month'].

    Returns:
    pd.DataFrame: DataFrame with a MultiIndex ['wyear', 'month', 'day'] or ['wyear', 'month'].
    """
    # Check if the index is a MultiIndex
    if not isinstance(dataset.index, pd.MultiIndex):
        print('Dataset given does not have a MultiIndex. Cannot convert.')
        return None

    # Check if the index names are either ['year', 'month', 'day'] or ['year', 'month']
    index_names = dataset.index.names
    if index_names not in [('year', 'month', 'day'), ('year', 'month')]:
        print('Dataset given does not have proper index (year, month, day OR year, month). Cannot convert.')
        return None

    # Reset index to work with columns
    dataset_reset = dataset.reset_index()

    # Calculate wyear using vectorized operations
    dataset_reset['wyear'] = dataset_reset['year'] + (dataset_reset['month'] >= 10).astype(int)

    # Set the new MultiIndex
    if index_names == ('year', 'month', 'day'):
        dataset_wyear = dataset_reset.set_index(['wyear', 'month', 'day'])
    else:
        dataset_wyear = dataset_reset.set_index(['wyear', 'month'])

    # Drop the original 'year' column
    dataset_wyear.drop(columns=['year'], inplace=True)
    return dataset_wyear

def convert_to_calendar_years(dataset):
    """
    Convert a pandas DataFrame with a MultiIndex ['wyear', 'month', 'day'] or ['wyear', 'month']
    to a DataFrame with a MultiIndex ['year', 'month', 'day'] or ['year', 'month'].

    Parameters:
    dataset (pd.DataFrame): Input DataFrame with a MultiIndex ['wyear', 'month', 'day'] or ['wyear', 'month'].

    Returns:
    pd.DataFrame: DataFrame with a MultiIndex ['year', 'month', 'day'] or ['year', 'month'].
    """
    # Check if the index is a MultiIndex
    if not isinstance(dataset.index, pd.MultiIndex):
        print('Dataset given does not have a MultiIndex. Cannot convert.')
        return None

    # Check if the index names are either ['wyear', 'month', 'day'] or ['wyear', 'month']
    index_names = dataset.index.names
    if index_names not in [('wyear', 'month', 'day'), ('wyear', 'month')]:
        print('Dataset given does not have proper index (wyear, month, day OR wyear, month). Cannot convert.')
        return None

    # Reset index to work with columns
    dataset_reset = dataset.reset_index()

    # Calculate calendar year using vectorized operations
    # If month >= 10, calendar year = wyear - 1, else calendar year = wyear
    dataset_reset['year'] = dataset_reset['wyear'] - (dataset_reset['month'] >= 10).astype(int)

    # Set the new MultiIndex
    if index_names == ('wyear', 'month', 'day'):
        dataset_year = dataset_reset.set_index(['year', 'month', 'day'])
    else:
        dataset_year = dataset_reset.set_index(['year', 'month'])

    # Drop the original 'wyear' column
    dataset_year.drop(columns=['wyear'], inplace=True)
    return dataset_year


def combine_augmentedWT(data_wyear, augmentedWT_data, RMSE_WT_data):
    """
    Combines observed with calculate WT data 
    
    WT_data: observed water year original WT data
    augmentedWT_data: calculated data from PUBS (calendar year)
    RMSE_WT_data: RMSE of calculated data from PUBS (calendar year)
    """
    huc_ids_to_use = RMSE_WT_data[RMSE_WT_data['rmse']<2]['huc_id'].values
    mean_WT_to_use  = augmentedWT_data[augmentedWT_data['huc_id'].isin(huc_ids_to_use)][['Date', 'huc_id', 'wtemp_predicted_lstm', 'wtemp_actual']]

    # separate to predicted and actual
    mean_WT_predict = mean_WT_to_use.pivot(index="Date", columns="huc_id",values="wtemp_predicted_lstm")
    mean_WT_actual = mean_WT_to_use.pivot(index="Date", columns="huc_id",values="wtemp_actual")

    # use actual values where you can, and predicted where data is missing
    WT_predict_actual_combined = mean_WT_actual.combine_first(mean_WT_predict)

    ### combine Augmented WT with exisiting data:
    ## prep data to look like exisiting WT data
    WT_predict_actual_combined.index = pd.to_datetime(WT_predict_actual_combined.index)
    WT_predict_actual_working = split_datetime(WT_predict_actual_combined)
    WT_predict_actual_working.columns.name = None
    WT_predict_actual_wyear = convert_to_water_years(WT_predict_actual_working)

    # use .combine_first
    # first keep all rows in data_wyear, then append any new ones from WT_predict_actual_wyear
    new_rows = WT_predict_actual_wyear.index.difference(data_wyear.index)
    desired_idx = data_wyear.index.append(new_rows)

    WT_combined = (
        data_wyear
        .combine_first(WT_predict_actual_wyear)
        .reindex(desired_idx)
    )
    print("Finished combining augmented WT data with original.")

    return WT_combined


def make_var_dfs(data, metadata, start_date, end_date, augmentedWT_data, RMSE_WT_data):
    """
    Process BASIN3d data and metadata for each variable.
    
    Parameters:
    data (pd.DataFrame): Data portion of BASIN3d file
    metadata (pd.DataFrame): Metadata portion of BASIN3d file
    start_date (str): Start date for filtering data
    end_date (str): End date for filtering data
    augmentedWT_data (pd.DataFrame): Augmented water temperature data
    RMSE_WT_data (pd.DataFrame): RMSE data for water temperature
    
    Returns:
    tuple: Lists of processed data and metadata DataFrames for each variable
    """
    list_data_dfs = []
    list_metadata_dfs = []
    
    for var in var_names:
        print(f"Processing {var} data...")
        
        # STEP 1: Filter data and metadata for the current variable
        var_metadata = metadata[metadata.index.str.contains(var)]
        var_data = data.loc[start_date:end_date, data.columns.str.contains(var)]
        
        # STEP 2: Separate by Mean, Min, Max
        data_min, data_mean, data_max = sep_min_mean_max(var_data, var)
        metadata_min, metadata_mean, metadata_max = sep_min_mean_max(var_metadata.transpose(), var)
        
        # Take MIN/MAX average, drop sites that overlap between min/max and mean, and add to MEAN dataset
        sites_min_max = list(set(data_min.columns).intersection(set(data_max.columns)))
        avg = (data_min[sites_min_max] + data_max[sites_min_max]) / 2
        
        # Filter out sites that already exist in MEAN
        avg_filtered = avg.drop(list(set(data_mean.columns).intersection(set(avg.columns))), axis=1)
        
        # Combine MEAN data with filtered average
        data_mean_plus = pd.concat([data_mean, avg_filtered], axis=1)
        
        # Update metadata
        intersection = list(set(metadata_mean.columns).intersection(set(metadata_min.columns)))
        metadata_min_filtered = metadata_min.drop(intersection, axis=1)
        selected_metadata = pd.concat([metadata_mean, metadata_min_filtered], axis=1)
        
        # STEP 3: Split datetime to days, months, years
        data_mean_plus = split_datetime(data_mean_plus)
        
        # STEP 4: Check for ICE and -9999
        if var == 'RDC':
            if data_mean_plus.map(lambda x: isinstance(x, str)).any().any():
                print("RDC data has 'ICE'. Please inspect (and potentially write function to eliminate).")
            data_mean_plus = data_mean_plus.map(lambda x: np.nan if x < 0 else x)
        else:
            data_mean_plus.replace(-999999.0, np.nan, inplace=True)
        
        # STEP 5: Regroup to water years
        data_wyear = convert_to_water_years(data_mean_plus)
        
        # STEP 6: Ensure site_ids in columns are integers
        data_wyear.columns = data_wyear.columns.astype('int')
        selected_metadata.columns = selected_metadata.columns.astype('int')
        
        # STEP 7: Add augmented WT data if applicable
        if var == 'WT':
            data_wyear = combine_augmentedWT(data_wyear, augmentedWT_data, RMSE_WT_data)
            print("Finished adding augmented WT data.")
        
        # Add to result lists
        list_data_dfs.append(data_wyear)
        list_metadata_dfs.append(selected_metadata)
    
    print("Finished processing all variables:")
    print("- Separated data and metadata by variable")
    print("- Extracted daily Mean and calculated Min/Max average")
    print("- Checked for ICE and -9999 values")
    print("- Regrouped to water years")
    print('-' * 42)
    return list_data_dfs, list_metadata_dfs



def apply_criteria_get_avail(list_data_dfs):
    """
    Apply data availability criteria to a list of DataFrames.
    Criteria: At least 10 days of data per month, and at least 11 months per year
    
    Parameters:
    list_data_dfs (list): List of DataFrames with water year data for [RDC, WT, SC] variables
    
    Returns:
    tuple: Lists of availability and percent availability DataFrames for each variable
    """
    wyears = np.arange(1951, 2023, 1)
    required_days_per_month = 10
    max_missing_days_per_month = 30 - required_days_per_month
    max_missing_months_per_year = 1  # Only 1 month can be missing (11 months required)
    
    list_avail_dfs = []
    list_peravail_dfs = []
    
    for var_idx, var in enumerate(var_names):
        print(f"Processing availability for {var}...")
        
        # Get the current variable's DataFrame
        var_data = list_data_dfs[var_idx]
        
        # Initialize availability and percent availability DataFrames
        availability_df = pd.DataFrame(0, index=wyears, columns=var_data.columns)
        percent_avail_df = pd.DataFrame(0.0, index=wyears, columns=var_data.columns)
        
        # Process each site
        for site in var_data.columns:
            site_data = var_data[site].to_frame()
            
            # Process each year
            for year in wyears:
                try:
                    year_data = site_data.loc[year]
                    
                    # Calculate percent availability (for visualization)
                    missing_days = year_data.isna().sum().item()
                    percent_avail_df.at[year, site] = (365 - missing_days) / 365
                    
                    # Fast path: If the whole year is missing fewer days than our threshold,
                    # mark it as available without checking each month
                    if missing_days <= max_missing_days_per_month:
                        availability_df.at[year, site] = 1
                        continue
                    
                    # Count months with insufficient data
                    insufficient_months = 0
                    
                    # Check each month
                    for month in pd.unique(year_data.index.get_level_values(0)):
                        month_data = year_data.loc[month]
                        missing_days_in_month = month_data.isna().sum().item()
                        
                        if missing_days_in_month > max_missing_days_per_month:
                            insufficient_months += 1
                            
                        # Early exit if we've already found too many insufficient months
                        if insufficient_months > max_missing_months_per_year:
                            break
                    
                    # Mark year as available if it meets our criteria
                    if insufficient_months <= max_missing_months_per_year:
                        availability_df.at[year, site] = 1
                
                except KeyError:
                    # Year not in the data, leave availability as 0
                    continue
        
        list_avail_dfs.append(availability_df)
        list_peravail_dfs.append(percent_avail_df)
    
    print("Finished applying criteria: 10 days per month, 11 months per year")
    print('-' * 42)
    return list_avail_dfs, list_peravail_dfs



def delete_save_sites(data_dir, list_avail_dfs, list_data_dfs, list_metadata_dfs):
    """
    Filter and save data by removing sites with no availability and clearing data 
    from years that do not meet criteria.
    
    Parameters:
    data_dir (str): Directory path to save the filtered data
    list_avail_dfs (list): List of availability DataFrames for each variable
    list_data_dfs (list): List of data DataFrames for each variable
    list_metadata_dfs (list): List of metadata DataFrames for each variable
    
    Returns:
    tuple: Lists of filtered data, metadata, and availability DataFrames
    """
    list_data_filtered_dfs = []
    list_metadata_filtered_dfs = []
    list_avail_filtered_dfs = []
    wyears = np.arange(1951, 2023, 1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(data_dir, 'Water_year')
    os.makedirs(output_dir, exist_ok=True)
    
    for var_idx, var in enumerate(var_names):
        print(f"Processing {var} data...")
        
        # Get copies of the current variable's DataFrames
        data_df = list_data_dfs[var_idx].copy()
        metadata_df = list_metadata_dfs[var_idx].copy()
        avail_df = list_avail_dfs[var_idx].copy()
        
        # Get sites with zero availability
        zero_avail_sites = avail_df.columns[avail_df.mean() == 0]
        
        # Remove sites with zero availability
        if len(zero_avail_sites) > 0:
            print(f"  Removing {len(zero_avail_sites)} sites with no availability")
            avail_df = avail_df.drop(zero_avail_sites, axis=1)
            data_df = data_df.drop(zero_avail_sites, axis=1)
            metadata_df = metadata_df.drop(zero_avail_sites, axis=1)
            print(f"  Sites with any availability: {len(avail_df.columns)}")
        
        # For remaining sites, set data to NaN for years that don't meet criteria
        for site in avail_df.columns:
            # Find years where availability is 0
            unavailable_years = wyears[avail_df[site] == 0]
            
            if len(unavailable_years) > 0:
                # Create a mask for the MultiIndex to identify rows to set to NaN
                mask = data_df.index.get_level_values(0).isin(unavailable_years)
                
                # Use vectorized operation to set values to NaN
                data_df.loc[mask, site] = np.nan
        
        # Append to result lists
        list_data_filtered_dfs.append(data_df)
        list_metadata_filtered_dfs.append(metadata_df)
        list_avail_filtered_dfs.append(avail_df)
        
        # Save to CSV files
        data_df.to_csv(os.path.join(output_dir, f'{var}_semicleaned_wy.csv'))
        metadata_df.to_csv(os.path.join(output_dir, f'{var}_metadata_wy.csv'))
        avail_df.to_csv(os.path.join(output_dir, f'{var}_availability_wy.csv'))
        
        print(f"  Saved {var} data files")
    
    print("Finished processing all variables:")
    print("- Removed sites with no availability")
    print("- Cleared data from years that do not meet criteria")
    print(f"- All data saved to {output_dir}/")
    print('-' * 42)
    
    return list_data_filtered_dfs, list_metadata_filtered_dfs, list_avail_filtered_dfs


def split_met_data(raw_met_data,MET_dir=None):
    """
    Splits the raw meteorologic data obtained from Google Earth Engine to separate dataframes

    Parameters:
    :raw_met_data: pandas dataframe of MET data from Google Earth Engine, PRISM temp and precip
    
    Returns 2 pandas dataframes for temp and precip with ['year','month'] as index, and site_ids (int) as columns
    """
    # Split into tmean DataFrame
    tmean_df = raw_met_data.filter(like='tmean').copy()

    # Split into ppt DataFrame
    ppt_df = raw_met_data.filter(like='ppt').copy()

    # Extract year and month columns, assuming 'date_agg' has datetime format:
    raw_met_data['year'] = pd.to_datetime(raw_met_data['date_agg']).dt.year
    raw_met_data['month'] = pd.to_datetime(raw_met_data['date_agg']).dt.month

    # Add year and month
    tmean_df['year'] = raw_met_data['year']
    tmean_df['month'] = raw_met_data['month']
    ppt_df['year'] = raw_met_data['year']
    ppt_df['month'] = raw_met_data['month']

    # Rename the columns by removing '|tmean' or '|ppt'
    tmean_df.columns = tmean_df.columns.str.replace(r'\|\w+', '', regex=True)
    ppt_df.columns = ppt_df.columns.str.replace(r'\|\w+', '', regex=True)

    #set index
    tmean_df_index = tmean_df.set_index(['year','month'])
    ppt_df_index = ppt_df.set_index(['year','month'])
    tmean_df_index.columns = tmean_df_index.columns.astype('int')
    ppt_df_index.columns = ppt_df_index.columns.astype('int')
    print("Finished splitting temp and precip.") 
    if MET_dir != None:
        tmean_df_index.to_csv(MET_dir /'temp_raw_2024_09_01.csv')
        ppt_df_index.to_csv(MET_dir / 'precip_raw_2024_09_01.csv')
        print(f"- All data saved to {MET_dir}/")
    print('-' * 42)
    return tmean_df_index, ppt_df_index
  


def nlcd_processing(nlcd_dir, output_dir=None, start_year=None, end_year=None):
    """
    Process National Land Cover Database (NLCD) data for counties in the Upper Colorado River Basin.
    
    Parameters:
    nlcd_dir (str): Path to directory containing NLCD CSV files for each county
    output_dir (str, optional): Path to save processed CSVs. If None, files are not saved.
    start_year (int): Beginning year of the study period
    end_year (int): End year of the study period
    
    Returns:
    pd.DataFrame: DataFrame containing NLCD percent cover change between start_year and end_year
    
    Notes:
    - CSV files should be named with state prefix (e.g., 'CO_CountyName.csv')
    - Each CSV should have a 'Period' column as the index
    """
    if start_year is None or end_year is None:
        raise ValueError("Both start_year and end_year must be provided")
    
    states = ['AZ', 'CO', 'NM', 'UT', 'WY']
    all_years = np.arange(1985, 2024, 1)
    
    # Dictionary to store DataFrames for each year
    year_dfs = {year: pd.DataFrame() for year in all_years}
    
    print(f"Processing NLCD data for counties in {', '.join(states)}...")
    
    # Process each state
    for state in states:
        csv_pattern = os.path.join(nlcd_dir, f"{state}*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"Warning: No CSV files found for state {state}")
            continue
        
        print(f"Found {len(csv_files)} counties for {state}")
        
        # Process each county file
        for file_path in csv_files:
            county_name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                # Read county data
                county_df = pd.read_csv(file_path, index_col='Period')
                
                # Extract data for each year and add to the corresponding year DataFrame
                for year in all_years:
                    # Find column that contains the year
                    year_cols = [col for col in county_df.columns if str(year) in col]
                    
                    if year_cols:
                        year_dfs[year][county_name] = county_df[year_cols[0]]
                    else:
                        print(f"Warning: No data for year {year} in {county_name}")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Create transposed DataFrames with totals for each year
    transposed_dfs = {}
    for year in all_years:
        if not year_dfs[year].empty:
            transposed_df = year_dfs[year].transpose()
            transposed_df['TOTAL'] = transposed_df.sum(axis=1)
            transposed_dfs[year] = transposed_df
    
    # Calculate change between start and end years
    if start_year in transposed_dfs and end_year in transposed_dfs:
        overall_change = transposed_dfs[end_year] - transposed_dfs[start_year]
        print(f"NLCD change calculated for period {start_year} to {end_year}")
    else:
        missing_years = []
        if start_year not in transposed_dfs:
            missing_years.append(start_year)
        if end_year not in transposed_dfs:
            missing_years.append(end_year)
        error_msg = f"Cannot calculate change: missing data for years {missing_years}"
        print(f"Error: {error_msg}")
        raise ValueError(error_msg)
    
    # Save results if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data for each year
        for year, df in transposed_dfs.items():
            output_path = os.path.join(output_dir, f'UCRB_NLCD_percentcover_counties_{year}.csv')
            df.to_csv(output_path)
        
        # Save change data
        change_path = os.path.join(output_dir, f'UCRB_NLCD_percentcover_counties_CHANGE{end_year}_{start_year}.csv')
        overall_change.to_csv(change_path)
        
        print(f"NLCD data saved to {output_dir}")
    print('-' * 42)
    return overall_change


def calculate_pet(tmean, gages_latitudes, output_dir=None, stop_year=2023):
    """
    Calculate Potential Evapotranspiration (PET) using the Thornthwaite method.
    
    Parameters:
    tmean (pd.DataFrame): Temperature DataFrame with MultiIndex ['year','month'] and site IDs as columns
    gages_latitudes (pd.DataFrame): DataFrame with site IDs as index and latitude information
    output_dir (str, optional): Directory to save the output CSV file. If None, file is not saved.
    stop_year (int, optional): Year to stop calculations at (inclusive). Defaults to 2023.
    
    Returns:
    pd.DataFrame: PET data with MultiIndex ['year','month'] and site IDs as columns
    """
    # Convert column names to integers for consistent comparison
    tmean_df = tmean.copy()
    tmean_df.columns = tmean_df.columns.astype(int)
    tmean_df = tmean_df[tmean_df.index.get_level_values('year') != stop_year+1]
    
    # Find sites that exist in both temperature data and gages data
    met_sites = set(tmean_df.columns)
    gage_sites = set(gages_latitudes.index.values)
    common_sites = list(met_sites.intersection(gage_sites))
    
    if not common_sites:
        raise ValueError("No common sites found between temperature data and gages data")
    
    # Report sites that are in temperature data but not in gages data
    missing_sites = met_sites - set(common_sites)
    if missing_sites:
        print(f'MET sites not in GAGESII: {missing_sites}')
    
    # Extract relevant latitude data
    site_latitudes = gages_latitudes.loc[common_sites, 'LAT_GAGE'].to_dict()
    
    # Initialize PET DataFrame with same structure as temperature data
    pet_data = pd.DataFrame(index=tmean_df.index, columns=common_sites)
    
    # Get unique years from the index, limited by stop_year
    years = [year for year in tmean_df.index.get_level_values(0).unique() if year <= stop_year]
    
    # Calculate PET for each site
    for site in common_sites:
        # Convert latitude to radians
        latitude_rad = pyeto.deg2rad(site_latitudes[site])
        
        for year in years:
            try:
                # Get temperature data for the current year and site
                temps = tmean_df.xs(year, level=0)[site].values
                
                # Calculate daylight hours
                daylight_hours = pyeto.monthly_mean_daylight_hours(latitude_rad, year)
                
                # Calculate PET values
                pet_values = pyeto.thornthwaite(temps, daylight_hours, year)
                
                # Assign PET values to the DataFrame
                for month, pet_value in enumerate(pet_values, 1):
                    pet_data.loc[(year, month), site] = pet_value
                    
            except Exception as e:
                print(f"Error calculating PET for site {site}, year {year}: {e}")
                # Fill with NaN for this year
                for month in range(1, 13):
                    pet_data.loc[(year, month), site] = np.nan
    
    # Ensure column names are integers
    pet_data.columns = pet_data.columns.astype(int)
    print('PET calculation completed.')
    
    # Save to file if output directory is provided
    if output_dir is not None:
        output_path = os.path.join(output_dir, 'pet_calc_thorn.csv')
        pet_data.to_csv(output_path)
        print(f'PET data saved to {output_path}')
    print('-' * 42)
    return pet_data

# calculate basin avg
def basin_averaging(variable, gagesii_int):
    '''
    Add a column for the basin averaged version (divide by basin area)
    sites are in columns, dates in index (monthly)

    Parameters:
    variable: dataframe with the observations for a varaiable with catchments as columns and dates as index
    gagesii_int: GAGESII traits to grab basin area
    
    Returns dataframe with column for basin average
    '''
    variable_wAVG = variable.copy()
    variable_wAVG['BASIN_AVG'] = 0.0
    gage_sizes = gagesii_int[['DRAIN_SQKM']].transpose()
    total_weights = gage_sizes.sum(axis=1).values[0]

    numerator = 0
    denominator = total_weights
    for index in range(0,len(variable.index)):
        month_values = variable.iloc[index]
        for site in variable.columns:
            numerator += month_values[site]*gage_sizes[site]
        variable_wAVG.iat[index,len(variable_wAVG.columns)-1] = numerator / denominator
        numerator = 0
    return variable_wAVG

# calculate SPEI
# Reference: https://spei.csic.es/home.html#p7
def calc_SPEI(gagesii_data, precipitation, pet, output_dir=None, start_year=1998, end_year=2022):
    """
    Calculate Standardized Precipitation Evapotranspiration Index (SPEI) for multiple basins.
    
    Parameters:
    gagesii_data (pd.DataFrame): DataFrame containing basin metadata including 'DRAIN_SQKM'
    precipitation (pd.DataFrame): DataFrame containing precipitation data with MultiIndex (year, month)
    pet (pd.DataFrame): DataFrame containing potential evapotranspiration data with MultiIndex (year, month)
    output_dir (str, optional): Directory to save output files. If None, files are not saved.
    start_year (int): Start year for truncated SPEI calculation (default: 1998)
    end_year (int): End year for truncated SPEI calculation (default: 2022)
    
    Returns:
    tuple: (annual SPEI values, truncated annual SPEI values), both pandas dataframes
    """
    # Ensure consistent types and find common sites
    precipitation.columns = precipitation.columns.astype(int)
    pet.columns = pet.columns.astype(int)
    gagesii_data.index = gagesii_data.index.astype(int)
    
    common_sites = list(set(gagesii_data.index.values).intersection(precipitation.columns.values))
    
    precipitation = precipitation[common_sites]
    pet = pet[common_sites]
    gagesii_data = gagesii_data.loc[common_sites]

    print("Calculating SPEI (Standardized Precipitation Evapotranspiration Index)...")
    
    # Calculate basin-averaged precipitation and PET
    precip_basin_avg = basin_averaging(precipitation, gagesii_data)
    pet_basin_avg = basin_averaging(pet, gagesii_data)
    
    # Step 1: Calculate water deficit (D = P - PET)
    water_deficit = precip_basin_avg - pet_basin_avg
    print(f"Calculated water deficit (P - PET) for {len(water_deficit.columns)} basins")
    
    # Step 2: Calculate 12-month rolling mean of water deficit
    deficit_rolling_mean = pd.DataFrame()
    
    for basin in water_deficit.columns:
        basin_rolling = water_deficit[basin].rolling(12, center=True).mean().dropna()
        deficit_rolling_mean[basin] = basin_rolling
    
    print(f"Calculated 12-month rolling mean of water deficit")
    
    # Step 3: Calculate SPEI for each basin using Gringorten plotting position
    all_basin_spei = []
    
    for basin in deficit_rolling_mean.columns:
        # Sort values for ranking
        basin_data = deficit_rolling_mean[basin].to_frame()
        basin_ranked = basin_data.sort_values(by=basin, ascending=True).reset_index()
        basin_ranked.index = basin_ranked.index + 1  # Start index at 1
        
        # Calculate Gringorten plotting position
        n_observations = len(basin_ranked)
        basin_ranked['gringorten'] = [(i - 0.44) / (n_observations + 0.12) for i in basin_ranked.index]
        
        # Restore date index and sort chronologically
        basin_date_sorted = basin_ranked.set_index(['year', 'month']).sort_index(level=[0, 1])
        
        # Calculate SPEI (standardized Gringorten values)
        gringorten_values = basin_date_sorted['gringorten']
        basin_date_sorted['SPEI'] = ((gringorten_values - gringorten_values.mean()) / 
                                     gringorten_values.std())
        
        # Prepare for concatenation
        basin_spei = basin_date_sorted[['SPEI']].rename(columns={'SPEI': basin})
        all_basin_spei.append(basin_spei)
    
    # Combine all basin SPEI values
    combined_spei = pd.concat(all_basin_spei, axis=1)
    print(f"Calculated SPEI for {len(combined_spei.columns)} basins")
    
    # Convert to water years
    spei_water_years = convert_to_water_years(combined_spei)
    spei_water_years.reset_index(inplace=True)
    
    # Create truncated version for specified period
    truncated_spei = spei_water_years.loc[
        (spei_water_years['wyear'] >= start_year) & 
        (spei_water_years['wyear'] <= end_year)
    ]
    
    # Set MultiIndex for both DataFrames
    spei_water_years.set_index(['wyear', 'month'], inplace=True)
    truncated_spei.set_index(['wyear', 'month'], inplace=True)
    
    # Calculate annual averages
    annual_spei = spei_water_years.groupby(level=0).mean()
    truncated_annual_spei = truncated_spei.groupby(level=0).mean()
    
    print(f"Calculated annual SPEI values for water years {annual_spei.index.min()} to {annual_spei.index.max()}")
    print(f"Truncated annual SPEI covers water years {start_year} to {end_year}")
    
    # Save results if output directory is provided
    if output_dir is not None:
        # Save annual SPEI values
        annual_file = os.path.join(output_dir, 'ann_spei_wy.csv')
        annual_spei.to_csv(annual_file)
        
        # Save truncated annual SPEI values
        truncated_file = os.path.join(output_dir, 'TRUN_ann_spei_wy.csv')
        truncated_annual_spei.to_csv(truncated_file)
        print(f"SPEI data saved to {output_dir}")
    print('-' * 42)
    return annual_spei, truncated_annual_spei


# flow normalization
def q_normalization(gagesii_info, RDC_data):
    ## (1) make copies of RDC and prune sites to have RDC and gages data
    gage_sizes = gagesii_info[['DRAIN_SQKM']].transpose()
    RDC_gages_sites = list(set(RDC_data.columns.values).intersection(set(gage_sizes.columns.values)))
    # (2) divide RDC by catchment size for m/s
    RDC_gages_data_mdf = RDC_data.copy()[RDC_gages_sites]
    for site in RDC_gages_sites:    
        RDC_gages_data_mdf[site] = RDC_gages_data_mdf[site] / gage_sizes[site].values[0]
    return RDC_gages_data_mdf


def prep_rdc(RDC_data, gagesii):
    """Prepare and normalize RDC (streamflow) data by basin area."""
    RDC_data.columns = RDC_data.columns.astype('int')
    RDC_ready = q_normalization(gagesii, RDC_data)
    print("Normalized streamflow by basin area (m^3 / s km^2)")
    return RDC_ready


# convert RDC to runoff
def Q_to_ann_runoff(RDC_data, gagesii, RDC_dir):
    RDC_ready = prep_rdc(RDC_data, gagesii)
    
    print("Convert Q (m^3 / s km^2) to daily runoff (mm/year), then sum for the whole year.")
    #### first, convert m^3 to mm^3 and km^2 to mm^2
    ## 1 m^3 = 10^9 mm^3
    RDC_withsites_mm3 = RDC_ready * 1e9

    ## 1 km^2 = 10^12 mm^2
    RDC_withsites_mm3_mm2 = RDC_withsites_mm3 / 1e12

    ## 86400 sec = 1 day
    RDC_withsites_mm3_mm2_day = RDC_withsites_mm3_mm2 * 86400

    ## daily runoff mm is now annual sum of runoff mm
    runoff_ann = RDC_withsites_mm3_mm2_day.groupby(level=[0]).sum(min_count=1)

    ## save the runoff file
    if RDC_dir != None:
        runoff_ann.to_csv(RDC_dir /'annual_runoff.csv')
        print(f"- Annual Runoff data saved to {RDC_dir}/")
    print('-' * 42)
    return runoff_ann

def prep_P_ann(P_data,gagesii):
    """Prepare Precip data and sum to annual."""
    precip_wyear = convert_to_water_years(P_data)
    gagesii.index = gagesii.index.astype('int')
    MET_sites_in_gages = list(set(gagesii.index.values).intersection(precip_wyear.columns.values))
    precip_wyear = precip_wyear[MET_sites_in_gages]
    # get annual summed precip (mm)
    P_ann = precip_wyear.groupby(level=[0]).sum(min_count=1)
    return P_ann

# calculate annual runoff efficiency
def get_runoff_efficiency(runoff_ann, P_data,gagesii, RDC_dir):
    print("prep P_ann")
    P_ann = prep_P_ann(P_data,gagesii)

    print("isolate to 1998-2022")
    runoff_years = runoff_ann.loc[1998:2022]
    P_years = P_ann.loc[1998:2022]

    print("drop sites that are all nan")
    runoff_years = runoff_years.dropna(axis=1, how='all')
    P_years = P_years.dropna(axis=1, how='all')

    print("calculate runoff ratio (AKA runoff efficiency)")
    sites_to_runoff_P =  list(set(runoff_years.columns).intersection(set(P_years.columns)))
    runoff_efficiency = runoff_years[sites_to_runoff_P] / P_years[sites_to_runoff_P]
    
    ## save the runoff efficiency file
    if RDC_dir != None:
        runoff_efficiency.to_csv(RDC_dir /'annual_runoff_efficiency.csv')
        print(f"- Annual Runoff Efficiency data saved to {RDC_dir}/")
    print('-' * 42)    
    return runoff_efficiency


# preprocess reservoir storage data
def combine_reservoir_data(reservoir_input_dir, reservoir_output_dir):
    
    # Initialize a dictionary to hold DataFrames
    dataframes = []
    allsites = []

    # Loop through all files in the folder
    for filename in os.listdir(reservoir_input_dir):
        if filename.endswith('.csv'):
            # Create the full file path
            file_path = os.path.join(reservoir_input_dir, filename)
            # Load the CSV file into a DataFrame
            vars()[filename[:-4]+'_file'] = pd.read_csv(file_path)

            # Set 'datetime' as the index
            vars()[filename[:-4]+'_file'].set_index('datetime', inplace=True)
        
            # Rename the 'storage' column to the site name (filename without .csv)
            site_name = filename[:-4]
        
            vars()[filename[:-4]+'_file'].rename(columns={'storage': site_name}, inplace=True)
        
            # Store the DataFrame in the dictionary with the filename (without .csv) as the key
            dataframes.append(vars()[filename[:-4]+'_file'])
            allsites.append(filename[:-4])

    # Now you can access each DataFrame using its filename without the .csv extension
    combined_df = pd.concat(dataframes, axis=1)
    combined_df.sort_index(inplace=True)
    ## save the combined reservoir file
    if reservoir_output_dir != None:
        combined_df.to_csv(reservoir_output_dir /'UCRB_reservoir_storage.csv')
        print(f"- Reservoir Storage data saved to {reservoir_output_dir}/")
    print('-' * 42)
    return combined_df


def load_processed_data(dirs, var_names, MET_vars):
    """
    Load all preprocessed/saved outputs for use in calculations and plotting notebooks.

    Parameters:
    dirs (dict): Directory paths from get_dirs()
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    MET_vars (list): List of MET variable names e.g. ['precip', 'temp']

    Returns:
    gagesii, gage_loc, gagesii_traits, pet_data_df, TRUN_ann_spei_wyears,
    runoff_ann, runoff_efficiency_ann, reservoirs_data, reservoirs_metadata,
    data, meta, avail, met
    """
    # GAGESii 
    gagesii = pd.read_csv(dirs['gagesii_dir'] / 'gagesII_metadata.csv').set_index('siteid')
    gagesii.index = gagesii.index.astype(int)
    gage_loc = gagesii[['LAT_GAGE']]
    gagesii_traits = pd.read_csv(dirs['gagesii_dir'] / 'processed_dataset.csv').set_index('STAID')

    # PET 
    pet_data_df = pd.read_csv(dirs['met_data_dir'] / 'pet_calc_thorn.csv')
    pet_data_df.set_index(['year', 'month'], inplace=True)
    pet_data_df.columns = pet_data_df.columns.astype(int)

    # SPEI 
    TRUN_ann_spei_wyears = pd.read_csv(dirs['spei_data_dir'] / 'TRUN_ann_spei_wy.csv')
    TRUN_ann_spei_wyears.set_index('wyear', inplace=True)
    #TRUN_ann_spei_wyears_wo_basinavg.columns = TRUN_ann_spei_wyears_wo_basinavg.columns.astype(int)

    # Runoff 
    runoff_ann = pd.read_csv(dirs['rdc_wt_sc_data_dir'] / 'annual_runoff.csv')
    runoff_ann.set_index('wyear', inplace=True)
    runoff_ann.columns = runoff_ann.columns.astype(int)

    runoff_efficiency_ann = pd.read_csv(dirs['rdc_wt_sc_data_dir'] / 'annual_runoff_efficiency.csv')
    runoff_efficiency_ann.set_index('wyear', inplace=True)
    runoff_efficiency_ann.columns = runoff_efficiency_ann.columns.astype(int)

    # Reservoirs 
    reservoirs_data = pd.read_csv(dirs['reservoir_data_dir'] / 'UCRB_RESERVOIR_storage.csv')
    reservoirs_metadata = pd.read_csv(dirs['inputs_dir'] / 'RESERVOIR_ALL' / 'reservoirs_meta_NOTCLEAN.csv')

    # LSTM WT PUBS - load RMSE data to get locations of sites, want less than 2 for RMSE
    RMSE_WT_data  = pd.read_csv(dirs['wt_lstm_data_dir'] / "full_274attr_lstm_RMSEs_all_huc_14_sites.csv",index_col=0)
    PUBS_sites_all3 = RMSE_WT_data[RMSE_WT_data['rmse'] < 2]['huc_id'].values
    
    # RDC, WT, SC
    data = {}
    meta = {}
    avail = {}

    for var in var_names:
        wy_dir = dirs['rdc_wt_sc_data_dir'] / 'Water_year'

        data[var] = pd.read_csv(wy_dir / f'{var}_semicleaned_wy.csv')
        data[var].set_index(['wyear', 'month', 'day'], inplace=True)
        data[var].columns = data[var].columns.astype(int)

        meta[var] = pd.read_csv(wy_dir / f'{var}_metadata_wy.csv')
        meta[var].rename({'Unnamed: 0': 'siteid'}, axis=1, inplace=True)
        meta[var].set_index('siteid', inplace=True)
        meta[var].columns = meta[var].columns.astype(int)

        avail[var] = pd.read_csv(wy_dir / f'{var}_availability_wy.csv')
        avail[var].rename({'Unnamed: 0': 'wyear'}, axis=1, inplace=True)
        avail[var].set_index('wyear', inplace=True)
        avail[var].columns = avail[var].columns.astype(int)

    # MET
    met = {}
    for MET in MET_vars:
        met[MET] = pd.read_csv(dirs['met_data_dir'] / f'{MET}_raw_2024_09_01.csv')
        met[MET].set_index(['year', 'month'], inplace=True)
        met[MET].columns = met[MET].columns.astype(int)

    print("------ loaded GAGESii, PET, SPEI, Runoff, Runoff Efficiency, Reservoirs ------")
    print("------ loaded RDC, WT, SC, precip, temp ------")

    return (gagesii, gage_loc, gagesii_traits, pet_data_df, TRUN_ann_spei_wyears,
            runoff_ann, runoff_efficiency_ann, reservoirs_data, reservoirs_metadata, data, meta, avail, met,RMSE_WT_data,PUBS_sites_all3 )

def prep_reservoir_data(reservoirs_data, start_year=1998, end_year=2022):
    """
    Prepare reservoir data for Mann-Kendall test by converting to
    annual water year averages.

    Parameters:
    reservoirs_data (DataFrame): Raw reservoir storage data with datetime column
    start_year (int): Start year for filtering (default: 1998)
    end_year (int): End year for filtering (default: 2022)

    Returns:
    reservoirs_years (DataFrame): Annual water year averages filtered to start/end year
    """
    reservoirs_datetime = reservoirs_data.set_index(pd.to_datetime(reservoirs_data['datetime']))
    reservoirs_datesplit = split_datetime(reservoirs_datetime).drop(columns=['datetime'])
    reservoirs_wyearsdaily = convert_to_water_years(reservoirs_datesplit)

    reservoirs_ann = reservoirs_wyearsdaily.groupby(level=[0]).mean()
    reservoirs_years = reservoirs_ann.loc[start_year:end_year]
    reservoirs_years = reservoirs_years.dropna(how='all', axis=1)

    print("------ reservoir data averaged to annual water year for MK test ------")
    return reservoirs_years, reservoirs_wyearsdaily

def prep_regression_data(met_wyear, runoff_ann, allsites, start_year=1998, end_year=2022):
    """
    Prepare precipitation and runoff data for drought regression analysis.

    Parameters:
    met_wyear (dict): {MET: DataFrame} water year MET data
    runoff_ann (DataFrame): Annual runoff data with wyear as index
    allsites (dict): {var: list} union of valid sites per variable
    start_year (int): Start year for filtering (default: 1998)
    end_year (int): End year for filtering (default: 2022)

    Returns:
    runoff_years (DataFrame): Filtered and cleaned annual runoff data
    P_years (DataFrame): Filtered and cleaned annual precipitation data
    """
    P_ann = met_wyear['precip'].groupby(level=[0]).sum(min_count=1)

    runoff_years = runoff_ann.loc[start_year:end_year]
    P_years      = P_ann.loc[start_year:end_year]

    runoff_years = runoff_years.dropna(axis=1, how='all')[allsites['RDC']]
    P_years      = P_years.dropna(axis=1, how='all')[allsites['RDC']]

    print("------ prepped P and runoff for regression ------")
    return runoff_years, P_years

def extend_annual_medians(annual_medians, withsites, runoff_efficiency_ann,
                           reservoirs_wyearsdaily, allsites):
    """
    Extend the annual_medians dict with precip, temp, runoff efficiency,
    and reservoir entries needed for pre-post analysis.

    Parameters:
    annual_medians (dict): Existing {var: DataFrame} from build_annual_medians_by_rc()
    withsites (dict): {var: DataFrame} data filtered to sites of interest
    runoff_efficiency_ann (DataFrame): Annual runoff efficiency with wyear as index
    reservoirs_wyearsdaily (DataFrame): Daily reservoir data with water year index
    allsites (dict): {var: list} union of valid sites per variable

    Returns:
    annual_medians (dict): Updated with precip, temp, runoff_efficiency, reservoirs keys
    """
    annual_medians['precip']            = withsites['precip'].groupby(level=[0]).median()
    annual_medians['temp']              = withsites['temp'].groupby(level=[0]).median()
    annual_medians['runoff_efficiency'] = runoff_efficiency_ann[allsites['RDC']]
    annual_medians['reservoirs']        = reservoirs_wyearsdaily.groupby(level=[0]).median()

    return annual_medians

def prep_met_water_years(met, MET_vars, gagesii):
    """
    Convert MET data to water years and filter to sites present in GAGESii.

    Parameters:
    met (dict): {MET: DataFrame} meteorological data keyed by variable name
    MET_vars (list): List of MET variable names e.g. ['precip', 'temp']
    gagesii (DataFrame): GAGESii metadata with site IDs as index

    Returns:
    met_wyear (dict): {MET: DataFrame} water year converted MET data filtered to GAGESii sites
    list_met_wyears (list): List of DataFrames in same order as MET_vars
    """
    MET_sites_in_gages = list(set(gagesii.index.values).intersection(met['precip'].columns.values))

    met_wyear = {}
    list_met_wyears = []

    for MET in MET_vars:
        met_wyear[MET] = convert_to_water_years(met[MET])
        met_wyear[MET] = met_wyear[MET][MET_sites_in_gages]
        list_met_wyears.append(met_wyear[MET])

    return met_wyear, list_met_wyears


def load_colocation_data(MED_relchange_map, ann_MED_relchange, var_names, dr_names, all3vars):
    """
    Load and reconstruct co-located site relative change data for all drought episodes.

    Parameters:
    data_for_figures (Path): Path to FIGURE_data directory
    var_names (list): Variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names

    Returns:
    all3vars (dict): {dr: list of int} co-located site IDs per drought episode
    MED_relchange_map (dict): {var: DataFrame} median relative change with LAT/LON/CLASS
    ann_MED_relchange (dict): {var: {dr: Series}} annual median relative change per site
    all3_ann_MEDrelchange (dict): {dr: DataFrame} relative change for co-located sites only
    all3_ann_MEDrelchange_allsites (dict): {dr: DataFrame} reindexed to full site union
    all_sites (array): Union of all site IDs across drought episodes
    """
    # ── Reconstruct co-located relchange per drought ───────────────────────────
    all3_ann_MEDrelchange          = {}
    all3_ann_MEDrelchange_allsites = {}

    for dr in dr_names:
        cols = [ann_MED_relchange[var][dr].loc[all3vars[dr]] for var in var_names]
        df                          = pd.concat(cols, axis=1)
        df.columns                  = var_names
        all3_ann_MEDrelchange[dr]          = df
        all3_ann_MEDrelchange_allsites[dr] = df.copy()

    # ── Reindex to full site union ────────────────────────────────────────────
    full_idx = pd.Index([])
    for dr in dr_names:
        full_idx = full_idx.union(all3_ann_MEDrelchange_allsites[dr].index)

    for dr in dr_names:
        all3_ann_MEDrelchange_allsites[dr] = \
            all3_ann_MEDrelchange_allsites[dr].reindex(full_idx)

    all_sites = full_idx.values

    print("------ Co-location data loaded ------")
    return (all3_ann_MEDrelchange, all3_ann_MEDrelchange_allsites, all_sites)