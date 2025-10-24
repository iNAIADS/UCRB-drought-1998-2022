# load packages
import os
import glob
import h5py
import pandas as pd
from datetime import date, timedelta
import numpy as np
import warnings
import pyeto

var_names = ['RDC','WT','SC']
MET_vars = ['precip','temp']

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
            if data_mean_plus.applymap(lambda x: isinstance(x, str)).any().any():
                print("RDC data has 'ICE'. Please inspect (and potentially write function to eliminate).")
            data_mean_plus = data_mean_plus.applymap(lambda x: np.nan if x < 0 else x)
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
    
    return overall_change


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


    
