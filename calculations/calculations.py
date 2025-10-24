# load packages
import os
import glob
import h5py
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import pymannkendall as mk
from datetime import date, timedelta
import numpy as np
import warnings
import pyeto
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing

var_names = ['RDC','WT','SC']
MET_vars = ['precip','temp']


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
    variable_wAVG['BASIN_AVG'] = 0
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
    spei_water_years = preprocessing.convert_to_water_years(combined_spei)
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
    return annual_spei, truncated_annual_spei

    
# plotting SPEI
def plotting_spei(ann_spei_wy, gagesii_int, RDC_all3_allsites, spei_cmap, figure_dir=None,OPTION=False):
    '''
    Plot the SPEI for given sites in elevation order and get table of SPEI values
    OPTION: save spei table with elevations

    Parameters:
    ann_spei_wy: pandas dataframe with SPEI values for sites as columns, years as index
    gagesii_int: GAGESII data
    RDC_all3_allsites: list of sites to use for SPEI (streamflow sites that match data criteria and are present in analysis)
    spei_cmap: colormap for plotting
    figure_dir: directory to save the figure
    OPTION: save spei table with elevations 

    Returns: plot and optional table
    '''
    ##### add a bunch of columns to make basin avg wider
    ann_spei = ann_spei_wy.copy()
    ann_spei['Basin Average'] = ann_spei['BASIN_AVG']
    ann_spei['BASIN_AVG'] = np.nan
    ann_spei['Basin Avg'] = ann_spei['Basin Average']
    ann_spei['BASIN Avg'] = ann_spei['Basin Average']
    ann_spei['Basin AVG'] = ann_spei['Basin Average']
    
    #### Sort sites in by DESCENDING MEAN ELEVATION AND only grab  sites
    gages_elevation = gagesii_int[['ELEV_MEAN_M_BASIN']]
    total_size_all3 = len(RDC_all3_allsites) # flow sites
    size_plus5_all3 = total_size_all3+5
    gages_sort_elev_all3 = gages_elevation.transpose()[RDC_all3_allsites].sort_values(by='ELEV_MEAN_M_BASIN',axis=1,ascending=False)
    sorted_sites_all3 = np.concatenate((gages_sort_elev_all3.columns.values.reshape((1,total_size_all3)),ann_spei.columns[-5:].values.reshape((1,5))),axis=1)
    sorted_sites_2_all3 = sorted_sites_all3.reshape((size_plus5_all3,))
    sorted_sites_2_all3 = list(map(str, sorted_sites_2_all3))
    ann_spei_2plot = ann_spei[sorted_sites_2_all3]

    ## PLOT: SPEI figure
    fig, ax = plt.subplots(figsize=(45,10))
    fsize=12
    data = ann_spei_2plot

    # Add title to the Heat map
    title = f'SPEI Heat Map Descending Elevation (Water Years, n={len(RDC_all3_allsites)})'

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=20)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Use the heatmap function from the seaborn package
    sns.heatmap(data,annot=False,annot_kws={"size": 10}, fmt=".1f",cmap=spei_cmap,vmin=-1.5,vmax=1.5,center=0,ax=ax)
    for t in ax.texts:
        if -0.5 <= float(t.get_text()) <=0.5:
            t.set_text(t.get_text())  
        else:
            t.set_text('') # if not it sets an empty text

    ax.set_ylabel('',fontsize=1)
    for label in (ax.get_yticklabels()):
        label.set_fontsize(16)

    ax.xaxis.set_major_locator(plt.MaxNLocator(204))
    for label in (ax.get_xticklabels()):
        label.set_fontsize(0)

    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    if figure_dir != None:
        plt.savefig(figure_dir+'/ALL3_SPEI_wy_elevation_VARSITES_wlabels.svg',dpi=300)
    plt.show()

    if OPTION:
        ## Add elevation to the SPEI table to save
        gages_sort_elev_all3['Basin Average'] = gages_sort_elev_all3.loc['ELEV_MEAN_M_BASIN'].mean()
        spei_2save = ann_spei_2plot.copy().transpose()
        spei_2save['Elevation'] = gages_sort_elev_all3.transpose()['ELEV_MEAN_M_BASIN']
        spei_2save.to_csv(figure_dir+'spei_ann_wyElevation.csv')
        print('SPEI and elevation dataframe saved: '+figure_dir+'spei_ann_wyElevation.csv')
    return


# identify drought and reference years
def identify_years(annual_spei_data, drought_threshold=-1, reference_threshold=1, 
                   window_size=3, end_year=None):
    """
    Identify drought years and corresponding reference years based on SPEI values.
    
    Parameters:
    annual_spei_data (pd.DataFrame): DataFrame containing annual SPEI values with years as index
    drought_threshold (float): SPEI threshold below which a year is considered a drought year (default: -1)
    reference_threshold (float): Absolute SPEI threshold below which a year is considered a reference year (default: 1)
    window_size (int): Number of years before and after a drought event to consider (default: 3)
    end_year (int, optional): Last year in the dataset, used to prevent looking beyond available data
    
    Returns:
    tuple: (
        drought_years (list): List of individual drought years
        drought_events (list): List of drought event names (consecutive drought years are grouped)
        drought_years_by_event (list): List of lists containing drought years for each event
        reference_years_by_event (list): List of lists containing reference years for each event
        all_years_by_event (list): List of lists containing all years (drought + surrounding) for each event
    )
    """
    # Set end_year to the maximum year in the data if not provided
    if end_year is None:
        end_year = annual_spei_data.index.max()
    
    # Identify drought years (SPEI < drought_threshold)
    drought_years = annual_spei_data.loc[annual_spei_data['BASIN_AVG'] < drought_threshold].index.tolist()
    
    if not drought_years:
        print("No drought years found based on the specified threshold")
        return [], [], [], [], []
    
    print(f"Identified {len(drought_years)} drought years: {drought_years}")
    
    # Group consecutive drought years into events
    drought_events = []
    drought_years_by_event = []
    
    # Sort drought years to ensure they're in chronological order
    drought_years.sort()
    
    # Group consecutive years into drought events
    current_event = [drought_years[0]]
    current_event_name = str(drought_years[0])
    
    for i in range(1, len(drought_years)):
        if drought_years[i] == drought_years[i-1] + 1:
            # Consecutive year - add to current event
            current_event.append(drought_years[i])
            current_event_name = f"{current_event[0]}_{current_event[-1]}"
        else:
            # Non-consecutive year - start a new event
            drought_events.append(current_event_name)
            drought_years_by_event.append(current_event)
            
            current_event = [drought_years[i]]
            current_event_name = str(drought_years[i])
    
    # Add the last event
    drought_events.append(current_event_name)
    drought_years_by_event.append(current_event)
    
    print(f"Grouped into {len(drought_events)} drought events: {drought_events}")
    
    # For each drought event, identify surrounding years within the window
    reference_years_by_event = []
    all_years_by_event = []
    
    for event_years in drought_years_by_event:
        event_start = min(event_years)
        event_end = max(event_years)
        
        # Initialize lists for this event
        reference_years = []
        all_years = []
        
        # Check years before the event
        for year in range(event_start - window_size, event_start):
            if year < annual_spei_data.index.min():
                continue
                
            if -reference_threshold < annual_spei_data.at[year, 'BASIN_AVG'] < reference_threshold:
                reference_years.append(year)
                all_years.append(year)
        
        # Add the drought years themselves
        all_years.extend(event_years)
        
        # Check years after the event
        for year in range(event_end + 1, event_end + window_size + 1):
            if year > end_year:
                continue
                
            if -reference_threshold < annual_spei_data.at[year, 'BASIN_AVG'] < reference_threshold:
                reference_years.append(year)
                all_years.append(year)
        
        # Sort the years
        reference_years.sort()
        all_years.sort()
        
        reference_years_by_event.append(reference_years)
        all_years_by_event.append(all_years)
    return drought_years, drought_events, drought_years_by_event, reference_years_by_event, all_years_by_event


def rel_change_median_monthly(reference_data, drought_data, drought_episode):
    # Step 1: Calculate monthly medians
    # For single-year drought episodes, group by month; for multi-year episodes, group by month
    if drought_data.index.names == ['month', 'day']:
        # Single-year drought episodes: group by month (level 0)
        drought_monthly_median = drought_data.groupby(level=0, sort=False).median()
    else:
        # Multi-year drought episodes: group by month (level 1)
        drought_monthly_median = drought_data.groupby(level=[0,1], sort=False).median().groupby(level=1, sort=False).median()
    
    # Reference period monthly medians (always group by month)
    reference_monthly_median = reference_data.groupby(level=[0,1], sort=False).median().groupby(level=1, sort=False).median()

     # Calculate the difference between drought and reference
    monthly_difference = drought_monthly_median - reference_monthly_median
   
    # Calculate monthly relative change (%)
    # Create a copy of reference data with very small values instead of zeros to avoid division by zero
    reference_monthly_divisor = reference_monthly_median.copy().replace(0, 1e-10)
    
    # Calculate initial monthly relative change
    monthly_relative_change = (
        (drought_monthly_median - reference_monthly_median) / 
        reference_monthly_divisor.abs() * 100
    )
        
    return ( 
        monthly_relative_change,  
        drought_monthly_median, 
        reference_monthly_median,
        monthly_difference
    )


def rel_change_median_annual(reference_data, drought_data, drought_episode) :
    if drought_data.index.names == ['month', 'day']:
        # Single-year drought episodes
        drought_annual_median = drought_data.median()
    else:
        # Multi-year drought episodes
        drought_annual_median = drought_data.groupby(level=0).median().median(axis=0)
    
    # Calculate ref annual medians, Average the medians for each year for each site  - one annual value
    reference_annual_median = reference_data.groupby(level=0).median().median(axis=0)

    # Calculate the difference between drought and reference
    annual_difference = drought_annual_median - reference_annual_median
   
    # Calculate monthly relative change (%)
    # Create a copy of reference data with very small values instead of zeros to avoid division by zero
    reference_annual_divisor = reference_annual_median.copy().replace(0, 1e-10)

    #  Calculate annual relative change (%)
    annual_relative_change = (
        (drought_annual_median - reference_annual_median) / 
        reference_annual_divisor.abs() * 100
    )
      
    return (
        annual_relative_change,   
        drought_annual_median, 
        reference_annual_median,
        annual_difference
    )


def rel_change_mean_monthly(reference_data, drought_data, drought_episode):
    # Step 1: Calculate monthly medians
    # For single-year drought episodes, group by month; for multi-year episodes, group by month
    if drought_data.index.names == ['month', 'day']:
        # Single-year drought episodes: group by month (level 0)
        drought_monthly_median = drought_data.groupby(level=0, sort=False).median()
    else:
        # Multi-year drought episodes: group by month (level 1)
        drought_monthly_median = drought_data.groupby(level=[0,1], sort=False).median().groupby(level=1, sort=False).mean()
    
    # Reference period monthly medians (always group by month)
    reference_monthly_median = reference_data.groupby(level=[0,1], sort=False).median().groupby(level=1, sort=False).mean()

     # Calculate the difference between drought and reference
    monthly_difference = drought_monthly_median - reference_monthly_median
   
    # Calculate monthly relative change (%)
    # Create a copy of reference data with very small values instead of zeros to avoid division by zero
    reference_monthly_divisor = reference_monthly_median.copy().replace(0, 1e-10)
    
    # Calculate initial monthly relative change
    monthly_relative_change = (
        (drought_monthly_median - reference_monthly_median) / 
        reference_monthly_divisor.abs() * 100
    )
        
    return ( 
        monthly_relative_change,  
        drought_monthly_median, 
        reference_monthly_median,
        monthly_difference
    )

def rel_change_mean_annual(reference_data, drought_data, drought_episode) :
    if drought_data.index.names == ['month', 'day']:
        # Single-year drought episodes: group by month (level 0)
        drought_annual_median = drought_data.median()
    else:
        # Multi-year drought episodes: group by month (level 1)
        drought_annual_median = drought_data.groupby(level=0).median().mean(axis=0)
    
    # Calculate ref annual medians, Average the medians for each year for each site  - one annual value
    reference_annual_median = reference_data.groupby(level=0).median().mean(axis=0)

    # Calculate the difference between drought and reference
    annual_difference = drought_annual_median - reference_annual_median
   
    # Calculate monthly relative change (%)
    # Create a copy of reference data with very small values instead of zeros to avoid division by zero
    reference_annual_divisor = reference_annual_median.copy().replace(0, 1e-10)

    #  Calculate annual relative change (%)
    annual_relative_change = (
        (drought_annual_median - reference_annual_median) / 
        reference_annual_divisor.abs() * 100
    )
      
    return (
        annual_relative_change,   
        drought_annual_median, 
        reference_annual_median,
        annual_difference
    )
    


# # getting long term climatology
def calculate_meteorological_climatology(meteorological_data, common_sites, drought_years, 
                                         longterm_avg_start=1998, end_year=2022):
    """
    Calculate meteorological climatology statistics for different time periods and drought events.
    
    Parameters:
    meteorological_data (list): List of DataFrames containing meteorological variables data
    common_sites (list): List of site IDs that are common across all datasets
    drought_years (list): List of individual drought years
    longterm_avg_start (int): Start year for long-term average calculation (default: 2000)
    end_year (int): End year for calculations (default: 2022)
    
    Returns:
    dict: Dictionary containing:
        - Long-term basin averages for each variable
        - Site-specific data for each variable
        - Drought-year averages for each variable
    """
    
    # Initialize result dictionary
    result = {
        'long_term_basin_avg': {},
        'site_data': {},
        'drought_avg': {},
        'drought_avg_list': {}
    }
    
    # Process each meteorological variable
    for var_idx, variable in enumerate(MET_vars):
        # Get data for current variable
        var_data = meteorological_data[var_idx]
        
        # Filter for common sites across all datasets
        common_sites_data = var_data[list(set(var_data.columns) & set(common_sites))]
        
        # Store site-specific data
        result['site_data'][variable] = common_sites_data
        
        # Calculate average for each drought year
        drought_year_avgs = {}
        drought_avg_list = []
        
        for year in drought_years:
            # Extract data for the specific drought year
            if year in common_sites_data.index.get_level_values('wyear'):
                year_data = common_sites_data.xs(year, level='wyear')
                # Calculate monthly average across all sites for this year
                year_avg = year_data.mean(axis=1)
                drought_year_avgs[str(year)] = year_avg
                drought_avg_list.append(year_avg)
            else:
                print(f"Warning: Year {year} not found in {variable} data")
                drought_year_avgs[str(year)] = None
        
        # Handle multi-year drought events (assuming they're consecutive years)
        # Find consecutive years in drought_years
        for i in range(len(drought_years) - 1):
            if drought_years[i] + 1 == drought_years[i+1]:
                # Create a combined average for consecutive drought years
                year1 = str(drought_years[i])
                year2 = str(drought_years[i+1])
                combined_name = f"{year1}_{year2}"
                
                if drought_year_avgs[year1] is not None and drought_year_avgs[year2] is not None:
                    combined_avg = (drought_year_avgs[year1] + drought_year_avgs[year2]) / 2
                    drought_year_avgs[combined_name] = combined_avg
        
        # Calculate long-term average (from start year to end year)
        # Filter data for the long-term period
        longterm_data = common_sites_data[
            (common_sites_data.index.get_level_values('wyear') >= longterm_avg_start) &
            (
                (common_sites_data.index.get_level_values('wyear') <= end_year) | 
                (
                    (common_sites_data.index.get_level_values('wyear') == end_year + 1) & 
                    (common_sites_data.index.get_level_values('month') <= 9)
                )
            )
        ]
        
        # Calculate basin-wide average by month for the long-term period
        longterm_basin_avg = longterm_data.mean(axis=1).groupby(level=1, sort=False).mean()
        
        # Store results
        result['long_term_basin_avg'][variable] = longterm_basin_avg
        result['drought_avg'][variable] = drought_year_avgs
        result['drought_avg_list'][variable] = drought_avg_list
    
    # Return specific outputs to maintain backward compatibility
    return (
        result['long_term_basin_avg'].get('precip', None),
        result['long_term_basin_avg'].get('temp', None),
        result['site_data'].get('precip', None),
        result['site_data'].get('temp', None),
        result['drought_avg_list'].get('precip', []),
        result['drought_avg_list'].get('temp', [])
    )
    


def prep_mapping(mapping_data_list, metadata_filtered_dfs, gages_reference_data, all_var=True,huc_code='14'):
    """
    Prepares mapping data by adding geographic coordinates and reference site classification.
    
    Parameters:
    mapping_data_list : List of DataFrames containing relative change data for each variable
    metadata_filtered_dfs : List of DataFrames containing metadata for each variable, including lat/lon coordinates
    gages_reference_data : DataFrame containing GAGES-II reference site information
    huc_code : str, Hydrologic Unit Code to filter reference sites (default: '14' for Upper Colorado River Basin)
    
    Returns:
    list: List of DataFrames with enhanced mapping data including coordinates and site classification
    """    
    # Get reference sites for the specified HUC region
    reference_sites = set(gages_reference_data[
        (gages_reference_data['HUC02'] == huc_code) & 
        (gages_reference_data['CLASS'] == 'Ref')
    ].index)
    
    # Process each variable
    enhanced_mapping_data = []
    
    for var_idx, variable in enumerate(var_names):
        if not all_var:
            if var_idx != 0:
                return enhanced_mapping_data
            
        # Get the data for this variable
        var_mapping_data = mapping_data_list[var_idx].copy()
        var_metadata = metadata_filtered_dfs[var_idx]
        
        # Initialize coordinate columns
        var_mapping_data['LAT'] = 0.0
        var_mapping_data['LON'] = 0.0
        
        # Add coordinates from metadata
        for site in var_mapping_data.index:
            if site in var_metadata.columns:
                var_mapping_data.at[site, 'LAT'] = var_metadata.loc['sampling_feature_lat', site]
                var_mapping_data.at[site, 'LON'] = var_metadata.loc['sampling_feature_long', site]
        
        # Add reference site classification
        var_mapping_data['CLASS'] = 'Non-Ref'
        for site in var_mapping_data.index:
            if site in reference_sites:
                var_mapping_data.at[site, 'CLASS'] = 'Ref'
        
        # For water temperature (WT): handle sites missing coordinates
        if variable == 'WT':
            # Identify sites with missing coordinates
            missing_coord_sites = var_mapping_data[
                (var_mapping_data['LAT'] == 0) | 
                (var_mapping_data['LON'] == 0)
            ].index
            
            # Fill in missing coordinates from GAGES-II data
            for site in missing_coord_sites:
                if site in gages_reference_data.index:
                    var_mapping_data.at[site, 'LAT'] = gages_reference_data.at[site, 'LAT_GAGE']
                    var_mapping_data.at[site, 'LON'] = gages_reference_data.at[site, 'LNG_GAGE']
                else:
                    print(f"Warning: Site {site} not found in GAGES-II reference data.")
        
        # Ensure proper data types
        var_mapping_data['LAT'] = var_mapping_data['LAT'].astype(float)
        var_mapping_data['LON'] = var_mapping_data['LON'].astype(float)
        var_mapping_data['Relative Change (%)'] = var_mapping_data['Relative Change (%)'].astype(float)
        
        # Add to result list
        enhanced_mapping_data.append(var_mapping_data)
    
    return enhanced_mapping_data            


def identify_years_site(site):
    """
    identify start year, end year, and total years of data for given site (takes an annual series)
    
    Returns:
    first_year,last_year,length, non_nan (series without nans)
    """
    non_nan = site.dropna()
    first_year = non_nan.index.min()
    last_year = non_nan.index.max()
    length = len(non_nan)

    return first_year,last_year,length, non_nan


def mann_kendall_table(df,newdf):
    """
    runs MK test and gets table of results for each site,
    
    Parameters:
    df (pd.DataFrame): DataFrame with years as rows and sites as columns, annual series
    newdf (pd.DataFrame): DataFrame with sites as index and columns = ['startyr','endyr','totalyrs','trend','h','p','z','tau','s','var_s','slope','intercept']
    
    Returns:
    newdf: filled in after applying mk test and running identify_years_site
    """
    for site in df.columns:
        site_series = df[site]
        startyr,endyr,totalyrs,nonnan = identify_years_site(site_series)
        trend,h,p,z,tau,s,var_s,slope,intercept =  mk.original_test(nonnan)
        newdf.at[site,'startyr'] = startyr.astype('int')
        newdf.at[site,'endyr'] = endyr.astype('int')
        newdf.at[site,'totalyrs'] = totalyrs
        newdf.at[site,'trend'] = trend
        newdf.at[site,'h'] = h.astype('bool')
        newdf.at[site,'p'] = p.astype('float')
        newdf.at[site,'z'] = float(z)
        newdf.at[site,'tau'] = tau.astype('float')
        newdf.at[site,'s'] = s.astype('float')
        newdf.at[site,'var_s'] = float(var_s)
        newdf.at[site,'slope'] = slope.astype('float')
        newdf.at[site,'intercept'] = intercept.astype('float')
    return newdf



def boxcox_transform_dataframe(df):
    """
    Apply Box-Cox transform to each column of a DataFrame individually,
    ignoring NaNs and propagating them in the output.
    
    Parameters:
    df (pd.DataFrame): DataFrame with years as rows and sites as columns.
    
    Returns:
    pd.DataFrame: Box-Cox transformed DataFrame with NaNs propagated.
    dict: Dictionary of lambdas used for each column.
    """
    transformed_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    lambdas = {}

    for col in df.columns:
        series = df[col]

        # Drop NaNs to get valid data
        valid = series.dropna()
        
        if valid.empty:
            # If the entire column is NaN, just keep it as is
            transformed_df[col] = np.nan
            lambdas[col] = None
            continue

        # Check for positive values (Box-Cox requires strictly positive)
        min_val = valid.min()
        if min_val <= 0:
            # Shift data by abs(min_val) + small epsilon to make all values positive
            shift = abs(min_val) + 1e-6
            shifted = valid + shift
            transformed_data, lam = stats.boxcox(shifted)
        else:
            shift = 0
            transformed_data, lam = stats.boxcox(valid)

        # Create a full-length array with NaNs
        transformed_col = pd.Series(data=np.nan, index=series.index)

        # Assign transformed values back to the valid indices
        transformed_col.loc[valid.index] = transformed_data

        transformed_df[col] = transformed_col
        lambdas[col] = (lam, shift)  # Store lambda and shift for inverse transform if needed

    return transformed_df, lambdas


def lag1_autocorrelation_wallis(residuals):
    """
    Calculate lag-1 autocorrelation of residuals using Wallis & O'Connell (1972) eq. (3).
    residuals: model.resid
    Returns lag-1 autocorrelation coefficient.
    """
    x = residuals.dropna()
    n = len(x)
    if n < 2:
        return 0.0  # Not enough data

    x_mean = x.mean()
    numerator = 0.0
    denominator = 0.0
    for i in range(n - 1):
        numerator += (x.iloc[i] - x_mean) * (x.iloc[i + 1] - x_mean)
    for i in range(n):
        denominator += (x.iloc[i] - x_mean) ** 2

    if denominator == 0:
        return 0.0  # Avoid division by zero

    p = numerator / denominator
    return p

def prewhiten_series(series, rho):
    """
    Hahn (2002) transformation: (Quoted from Saft et al 2015)
    "all variables...each time step [are] reduced by the value of autocorrelation * variable at previous time step
    X'_t = X_t - rho * X_{t-1}
    For missing X_{t-1}, substitute mean of series.
    """
    X = series.copy()
    X_shifted = X.shift(1)
    # Replace NaNs in X_{t-1} with mean of series (excluding NaNs)
    mean_val = X.mean()
    X_shifted_filled = X_shifted.fillna(mean_val)
    X_pw = X - rho * X_shifted_filled
    return X_pw

def autocorrelation_corrected_regression(Q, P, I):
    """
    Perform autocorrelation correction and regression for one site/drought annually
    using Saft et al 2015: Wallis & O'Connell (1972) to get autocorrelation and Hahn (2002) to transform variables
    """
    df = pd.DataFrame({'Q': Q, 'P': P, 'I': I}).dropna(subset=['Q', 'P'])
    if len(df) < 5:
        return None # Not enough data

    # Initial regression
    X_init = df[['I', 'P']]
    X_init = sm.add_constant(X_init)
    y_init = df['Q']
    model_init = sm.OLS(y_init, X_init).fit()
    
    rho = lag1_autocorrelation_wallis(model_init.resid)   # autocorrelation of residuals
    
    # transform variables using autocorrelation
    Q_pw = prewhiten_series(df['Q'], rho)
    P_pw = prewhiten_series(df['P'], rho)
    I_pw = prewhiten_series(df['I'], rho)
    
    # drop any NaNs introduced
    df_pw = pd.DataFrame({'Q': Q_pw, 'P': P_pw, 'I': I_pw}).dropna()
    if len(df_pw) < 5:
        return None     # Not enough data
    
    # Re-fit regression on transformed data
    X_pw = df_pw[['I', 'P']]
    X_pw = sm.add_constant(X_pw)
    y_pw = df_pw['Q']
    model_pw = sm.OLS(y_pw, X_pw).fit()
    
    return model_pw, rho     # Return model results and autocorrelation


def run_drought_regressions(Q_all, P_all, drought_years):
    """
    Perform regression Q = a0 + a1*I + a2*P + epsilon for each site.
    Parameters:
    - Q_all, P_allt: DataFrames with annual runoff and precip (years x sites)
    - drought_years: list or set of drought years 
    Returns:
    - results_df: DataFrame with index=siteIDs and columns ['a0', 'a1', 'a2', 'pval_a1', 'n_points']
    - I_all: df signifying drought and nondrought
    """
    # drought indicator, same shape as Q_all
    I_all = pd.DataFrame(0, index=Q_all.index, columns=Q_all.columns)
    # indicator=1 for drought years 
    drought_years_set = set(drought_years)
    drought_years_in_data = [y for y in Q_all.index if y in drought_years_set]
    I_all.loc[drought_years_in_data, :] = 1
    
    sites = Q_all.columns
    results = []

    for site in sites:
        Q = Q_all[site]
        P = P_all[site]
        I = I_all[site]
        
        # Run autocorrelation-corrected regression
        result = autocorrelation_corrected_regression(Q, P, I)
        
        if result is None: # Not enough data or failed
            results.append({'site': site, 'a0': np.nan, 'a1': np.nan, 'a2': np.nan, 'pval_a1': np.nan, 'rho': np.nan})
            continue

        model_pw, rho = result
        a0 = model_pw.params['const']
        a1 = model_pw.params['I']
        a2 = model_pw.params['P']
        pval_a1 = model_pw.pvalues['I']

        results.append({'site': site, 'a0': a0, 'a1': a1, 'a2': a2, 'pval_a1': pval_a1, 'rho': rho})
        
    results_df = pd.DataFrame(results).set_index('site')
    return results_df, I_all

