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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import preprocessing

var_names = ['RDC','WT','SC']
MET_vars = ['precip','temp']
all_variables = var_names + MET_vars
all_vars_qp   = var_names + MET_vars + ['runoff_efficiency', 'reservoirs']
    

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


def build_drought_site_dicts(dr_names, list_all_years, list_ref_years, list_dr_years,
                              var_names, list_all_data, avail):
    """
    Build dictionaries organizing data by drought episode, variable, and year type.

    Parameters:
    dr_names (list): List of drought episode names e.g. ['2001_2002', '2012', '2018', '2020_2021']
    list_all_years (list): List of year lists for each drought episode (drought + reference years)
    list_ref_years (list): List of reference year lists for each drought episode
    list_dr_years (list): List of drought year lists for each drought episode
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    list_all_data (list): List of DataFrames corresponding to each variable in var_names
    avail (dict): Availability DataFrames keyed by variable name

    Returns:
    dr_all3, ref_all3, dr_years_list, sites_all3, refyrs_all3, dryrs_all3
    """
    # Drought episode year lists 
    dr_all3 = {}
    ref_all3 = {}
    dr_years_list = {}

    for number, drought in enumerate(dr_names):
        dr_all3[drought] = list_all_years[number]
        ref_all3[drought] = list_ref_years[number]
        dr_years_list[drought] = list_dr_years[number]

    # Sites with data for each drought episode 
    sites_all3 = {var: {} for var in var_names}

    for num, var in enumerate(var_names):
        for drought in dr_names:
            sites_all3[var][drought] = list_all_data[num].copy()

            for site in avail[var].columns:
                for year in dr_all3[drought]:
                    if avail[var].at[year, site] == 0:
                        if site in sites_all3[var][drought].columns.values:
                            sites_all3[var][drought].drop(site, axis=1, inplace=True)

    # Reference year slices
    refyrs_all3 = {var: {} for var in var_names}

    for dr in dr_names:
        for var in var_names:
            refyrs_all3[var][dr] = sites_all3[var][dr].transpose()[ref_all3[dr]].transpose()

    # Drought year slices 
    dryrs_all3 = {var: {} for var in var_names}

    dr_names_1yr = ['2012', '2018']
    for dr in dr_names_1yr:
        for var in var_names:
            dryrs_all3[var][dr] = sites_all3[var][dr].transpose()[int(dr)].transpose()

    for var in var_names:
        dryrs_all3[var]['2001_2002'] = sites_all3[var]['2001_2002'].transpose()[[2001, 2002]].transpose()
        dryrs_all3[var]['2020_2021'] = sites_all3[var]['2020_2021'].transpose()[[2020, 2021]].transpose()

    print("------ created framework of dictionaries for drought, reference, and all years ------")

    return dr_all3, ref_all3,dr_years_list, sites_all3, refyrs_all3, dryrs_all3


def build_site_unions(var_names, dr_names, sites_all3):
    """
    Build union of valid sites across drought episodes and variables.

    Parameters:
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): List of drought episode names
    sites_all3 (dict): sites_all3[var][drought] = DataFrame of valid sites

    Returns:
    allsites (dict): {var: list} union of valid sites across all drought episodes
    all3_all_sites (list): union of valid sites across all variables and drought episodes
    all3vars (dict): {drought: list} sites common to all variables for each drought episode
    """
    # ── Union of sites per variable across all drought episodes ────────────────
    allsites = {}
    print('Number of sites for each variable across all drought episodes:')
    for var in var_names:
        allsites[var] = []
        for dr in dr_names:
            allsites[var] = list(set(allsites[var]) | set(sites_all3[var][dr].columns.values))
        print(f'{var}: {len(allsites[var])}')

    # ── Union across all variables too ─────────────────────────────────────────
    all3_all_sites = list(set().union(*[allsites[var] for var in var_names]))
    print(f'Total unique sites across all variables and episodes: {len(all3_all_sites)}')

    # ── Sites common to all variables per drought episode ──────────────────────
    print('Number of sites with all variables for each drought episode:')
    all3vars = {}
    for dr in dr_names:
        var1and2 = set(sites_all3[var_names[0]][dr].columns.values).intersection(
                   set(sites_all3[var_names[1]][dr].columns.values))
        all3vars[dr] = list(var1and2.intersection(
                       set(sites_all3[var_names[2]][dr].columns.values)))
        print(f'{dr}: {len(all3vars[dr])}')

    return allsites, all3_all_sites, all3vars
    

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


def calculate_rel_change_all(var_list, dr_names, ref_all3, dr_years,
                              data_dict=None, refyrs_all3=None, dryrs_all3=None,
                              use_presliced=True):
    """
    Calculate annual and monthly median relative change for all variables and drought episodes.
    Can operate in two modes:
    - use_presliced=True  : uses refyrs_all3 and dryrs_all3 (pre-sliced by drought episode)
                            e.g. for RDC, WT, SC
    - use_presliced=False : slices from data_dict using ref_all3 and dr_years on wyear index
                            e.g. for MET variables (precip, temp)
    Parameters:
    var_list (list): List of variable names to process
    dr_names (list): List of drought episode names
    ref_all3 (dict): {drought: list} reference years per episode
    dr_years (dict): {drought: list} drought years per episode
    data_dict (dict): {var: DataFrame} used when use_presliced=False
    refyrs_all3 (dict): {var: {drought: DataFrame}} used when use_presliced=True
    dryrs_all3 (dict): {var: {drought: DataFrame}} used when use_presliced=True
    use_presliced (bool): Toggle between the two slicing modes (default: True)

    Returns:
    ann_MED (dict): {var: {drought: {'relchange', 'drought', 'ref', 'diff'}}}
    mon_MED (dict): {var: {drought: {'relchange', 'drought', 'ref', 'diff'}}}
    """
    ann_MED = {var: {} for var in var_list}
    mon_MED = {var: {} for var in var_list}

    for var in var_list:
        for dr in dr_names:
            ann_MED[var][dr] = {}
            mon_MED[var][dr] = {}

            if use_presliced:
                df_ref = refyrs_all3[var][dr]
                df_dr  = dryrs_all3[var][dr]
            else:
                df_ref = data_dict[var][data_dict[var].index.get_level_values('wyear').isin(ref_all3[dr])]
                df_dr  = data_dict[var][data_dict[var].index.get_level_values('wyear').isin(dr_years[dr])]

            (ann_MED[var][dr]['relchange'],ann_MED[var][dr]['drought'],ann_MED[var][dr]['ref'],ann_MED[var][dr]['diff'] ) = rel_change_median_annual(df_ref, df_dr, dr)

            (mon_MED[var][dr]['relchange'],mon_MED[var][dr]['drought'],mon_MED[var][dr]['ref'], mon_MED[var][dr]['diff']) = rel_change_median_monthly(df_ref, df_dr, dr)

    print(f"------ calculated annual and monthly median relative change for {var_list}  ------")
    return ann_MED, mon_MED

def aggregate_episodes(var_list, dr_names, ann_MED, include_dr_ref=True):
    """
    Concatenate relative change data across all drought episodes and average
    sites that appear in more than one episode.

    Parameters:
    var_list (list): List of variable names to process e.g. var_names or MET_vars
    dr_names (list): List of drought episode names
    ann_MED (dict): {var: {drought: {'relchange', 'drought', 'ref', ...}}}
                    as returned by calculate_rel_change_all
    include_dr_ref (bool): If True, also aggregates 'drought' and 'ref' keys
                           and renames columns. Use True for RDC/WT/SC,
                           False for MET variables (default: True)

    Returns:
    MED_allsites (dict): {var: {'relchange', ...}} concatenated across all episodes
    MED_all3 (dict):     {var: {'relchange', ...}} averaged across episodes per site
    """
    epis_lists   = {var: {'relchange': [], 'dr': [], 'ref': []} for var in var_list}
    MED_allsites = {var: {} for var in var_list}
    MED_all3     = {var: {} for var in var_list}

    for var in var_list:
        for dr in dr_names:
            if include_dr_ref:
                ann_MED[var][dr]['relchange'] = ann_MED[var][dr]['relchange'].to_frame().rename(columns={0: 'Relative Change (%)'})
                ann_MED[var][dr]['drought']   = ann_MED[var][dr]['drought'].to_frame().rename(columns={0: 'DR'})
                ann_MED[var][dr]['ref']        = ann_MED[var][dr]['ref'].to_frame().rename(columns={0: 'REF'})

                epis_lists[var]['dr'].append(ann_MED[var][dr]['drought'])
                epis_lists[var]['ref'].append(ann_MED[var][dr]['ref'])

            epis_lists[var]['relchange'].append(ann_MED[var][dr]['relchange'])

        # Concat and average across episodes
        MED_allsites[var]['relchange'] = pd.concat(epis_lists[var]['relchange'])
        MED_all3[var]['relchange']     = MED_allsites[var]['relchange'].groupby(level=0).mean()

        if include_dr_ref:
            MED_allsites[var]['dr']  = pd.concat(epis_lists[var]['dr'])
            MED_all3[var]['dr']      = MED_allsites[var]['dr'].groupby(level=0).mean()

            MED_allsites[var]['ref'] = pd.concat(epis_lists[var]['ref'])
            MED_all3[var]['ref']     = MED_allsites[var]['ref'].groupby(level=0).mean()

            print(f'Median Annual Relative Change: {var}', MED_all3[var]['relchange'].median().values)
            print(f'Median Annual DR:              {var}', MED_all3[var]['dr'].median().values)
            print(f'Median Annual Ref:             {var}', MED_all3[var]['ref'].median().values)
        else:
            print(f'Median Annual Relative Change: {var}', MED_all3[var]['relchange'].median())

    return MED_allsites, MED_all3
    
    
# # compute seasonal medians
def make_seasonal_dict(year,yespriorSept=False):
    if yespriorSept: # continuous fall from last water year
        seasonal_dict = {
            'SON': [(year-1, 9), (year, 10), (year, 11)],  # Previous year + September, current year
            'DJF': [(year, 12), (year, 1), (year, 2)],  
            'MAM': [(year, 3), (year, 4), (year, 5)],  
            'JJA': [(year, 6), (year, 7), (year, 8)]   
        }
    else: # one water year,  non-split FALL 
        seasonal_dict = {
            'OND': [(year, 10), (year, 11), (year, 12)],  
            'JFM': [(year, 1), (year, 2), (year, 3)],  
            'AMJ': [(year, 4), (year, 5), (year, 6)],  
            'JAS': [(year, 7), (year, 8), (year, 9)] 
        }
    return seasonal_dict


def compute_seasonal_medians(monthly_medians,year,yespriorSept=False):
    
    seasonal_dict = make_seasonal_dict(year,yespriorSept)
    
    allseasons = pd.DataFrame(index = monthly_medians.columns)
    
    for season, months in seasonal_dict.items():
        season_values = pd.DataFrame(index = monthly_medians.columns)
        
        for year, month in months:
            #print(year,month)
            if (year, month) in monthly_medians.index:
                season_values[month] = monthly_medians.loc[(year, month)]

        allseasons[season] = season_values.median(axis=1)
    return allseasons


def dr_ref_period_seasonal_medians(years, vartype, monthly_medians, yespriorSept, seasons):
    """
    Calculate seasonal medians for drought or reference years.

    Parameters:
    years (list): List of years for the period (drought or reference)
    vartype (str): Either 'dr' or 'ref'
    monthly_medians (DataFrame): Monthly median data with sites as columns
    yespriorSept (bool): Whether to include prior September
    seasons (list): List of season labels e.g. ['OND', 'JFM', 'AMJ', 'JAS']

    Returns:
    DataFrame: Seasonal medians with sites as index and seasons as columns
    """
    sites = monthly_medians.columns

    year_medians = {year: compute_seasonal_medians(monthly_medians, year, yespriorSept) for year in years}

    result = pd.DataFrame(index=sites, columns=seasons)

    if vartype == 'dr':
        if len(years) == 1:
            result = year_medians[years[0]]

        elif len(years) == 2:
            for season in seasons:
                result[season] = np.median([year_medians[years[0]][season],year_medians[years[1]][season]],axis=0)

    elif vartype == 'ref':
        for season in seasons:
            season_all_refyears = pd.DataFrame(index=sites)

            for year in years:
                season_all_refyears[year] = year_medians[year][season]

            result[season] = season_all_refyears.median(axis=1)

    else:
        raise ValueError(f"vartype must be 'dr' or 'ref', got '{vartype}'")

    return result


def calculate_seasonal_rel_change(var_list, dr_names, dr_years, ref_all3, seasons, vartypes,
                                   site_index_dict, yespriorSept=False,
                                   sites_all3=None, use_presliced=True):
    """
    Calculate seasonal median relative change across drought episodes for all variables.

    Can operate in two modes:
    - use_presliced=True  : groups sites_all3[var][dr] by wyear/month before passing
                            to dr_ref_period_seasonal_medians. Use for RDC, WT, SC.
    - use_presliced=False : passes data_dict[var] directly to dr_ref_period_seasonal_medians.
                            Use for MET variables where data is already at monthly resolution.

    Parameters:
    var_list (list): List of variable names to process
    dr_names (list): List of drought episode names
    dr_years (dict): {drought: list} drought years per episode
    ref_all3 (dict): {drought: list} reference years per episode
    seasons (list): List of season labels e.g. ['OND', 'JFM', 'AMJ', 'JAS']
    vartypes (list): List of value types e.g. ['dr', 'ref', 'relchange']
    site_index_dict (dict): {var: list or Index} sites to use as DataFrame index per variable.
                             For RDC/WT/SC pass allsites, for MET pass {MET: congruent_sites[MET].columns}
    yespriorSept (bool): Whether to include prior September (default: False)
    sites_all3 (dict): {var: {drought: DataFrame}} used when use_presliced=True
    use_presliced (bool): Toggle between the two modes (default: True)

    Returns:
    seasonMED (dict):      {var: {drought: {vartype: DataFrame}}}
    seasonMED_all (dict):  {var: {vartype: DataFrame}} averaged across episodes
    allep_seasonal (dict): {var: {vartype: DataFrame}} median across sites per season
    allep_combined (dict): {var: DataFrame} final concat of dr, ref, relchange
    """
    seasonMED      = {var: {dr: {} for dr in dr_names} for var in var_list}
    seasonMED_all  = {var: {} for var in var_list}
    allep_seasonal = {var: {} for var in var_list}
    allep_combined = {}

    for var in var_list:
        sites = site_index_dict[var]

        season_epis = {
            season: {vt: pd.DataFrame(index=sites) for vt in vartypes}
            for season in seasons
        }

        for dr in dr_names:
            if use_presliced:
                mon_data = sites_all3[var][dr].groupby(level=['wyear', 'month']).median()
            else:
                mon_data = site_index_dict['data'][var]
                df_ref = data_dict[var][data_dict[var].index.get_level_values('wyear').isin(ref_all3[dr])]
                df_dr  = data_dict[var][data_dict[var].index.get_level_values('wyear').isin(dr_years[dr])]

            # dr and ref seasonal medians
            seasonMED[var][dr]['dr'] = dr_ref_period_seasonal_medians(dr_years[dr], 'dr', mon_data, yespriorSept, seasons)

            seasonMED[var][dr]['ref'] = dr_ref_period_seasonal_medians(ref_all3[dr], 'ref', mon_data, yespriorSept, seasons)

            # relative change 
            reference_divisor = seasonMED[var][dr]['ref'].copy().replace(0, 1e-10)
            seasonMED[var][dr]['relchange'] = (
                (seasonMED[var][dr]['dr'] - reference_divisor) / reference_divisor.abs() * 100
            )

        # average across drought episodes 
        for vartype in vartypes:
            seasonMED_all[var][vartype] = pd.DataFrame(index=sites, columns=seasons)

            for season in seasons:
                for dr in dr_names:
                    season_epis[season][vartype][dr] = seasonMED[var][dr][vartype][season]

                seasonMED_all[var][vartype][season] = season_epis[season][vartype].mean(axis=1)

            allep_seasonal[var][vartype] = (
                seasonMED_all[var][vartype].median()
                .to_frame()
                .rename(columns={0: vartype})
            )

        allep_combined[var] = pd.concat([allep_seasonal[var][vt] for vt in vartypes], axis=1)
        print(f'{var} Median SEASONAL Relative Change:\n', allep_combined[var])

    return seasonMED, seasonMED_all, allep_seasonal, allep_combined


    
def calculate_longterm_avg(ready, allsites, var_names, longterm_avg_start=1998, end_year=2022):
    """
    Calculate long-term climatology averages for each variable.

    Parameters:
    ready (dict): {var: DataFrame} normalized/prepared data keyed by variable name
    allsites (dict): {var: list} union of valid sites per variable
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    longterm_avg_start (int): Start year for long-term average (default: 1998)
    end_year (int): End year for long-term average (default: 2022)

    Returns:
    longtermavg (dict): {var: DataFrame} pruned data for long-term period
    basin_avg (dict): {var: Series} monthly mean averaged across all sites
    """
    longtermavg = {}
    basin_avg = {}

    for var in var_names:
        longtermavg[var] = ready[var][allsites[var]]

        wyear = longtermavg[var].index.get_level_values('wyear')
        longtermavg[var] = longtermavg[var][(wyear >= longterm_avg_start) & (wyear <= end_year)]

        basin_avg[var] = longtermavg[var].groupby(level=1, sort=False).mean().mean(axis=1)

    return longtermavg, basin_avg


def build_site_dicts_no_filter(ref_all3, dr_names, list_all_years, list_ref_years, list_dr_years,
                                var_names, list_all_data):
    """
    Build drought/reference/site dictionaries without availability filtering.

    Parameters:
    ref_all3: list of lists of drought years per episode
    dr_names (list): Drought episode names
    list_all_years (list): All years per drought episode
    list_ref_years (list): Reference years per drought episode
    list_dr_years (list): Drought years per drought episode
    var_names (list): Variable names matching order of list_all_data
    list_all_data (list): DataFrames per variable

    Returns:
    sites_all3, refyrs_all3, dryrs_all3
    """

    # Full data copy per var/drought — no availability filtering
    sites_all3 = {var: {} for var in var_names}
    for num, var in enumerate(var_names):
        for drought in dr_names:
            sites_all3[var][drought] = list_all_data[num].copy()

    # Reference year slices
    refyrs_all3 = {var: {} for var in var_names}
    for dr in dr_names:
        for var in var_names:
            refyrs_all3[var][dr] = sites_all3[var][dr].transpose()[ref_all3[dr]].transpose()

    # Drought year slices
    dryrs_all3    = {var: {} for var in var_names}
    dr_names_1yr  = ['2012', '2018']

    for dr in dr_names_1yr:
        for var in var_names:
            dryrs_all3[var][dr] = sites_all3[var][dr].transpose()[int(dr)].transpose()

    for var in var_names:
        dryrs_all3[var]['2001_2002'] = sites_all3[var]['2001_2002'].transpose()[[2001, 2002]].transpose()
        dryrs_all3[var]['2020_2021'] = sites_all3[var]['2020_2021'].transpose()[[2020, 2021]].transpose()

    print("------ created site dictionaries (no availability filtering) ------")

    return sites_all3, refyrs_all3, dryrs_all3

# # getting long term climatology
def calculate_meteorological_climatology(meteorological_data, common_sites, drought_years,
                                         longterm_avg_start=1998, end_year=2022):
    """
    Calculate meteorological climatology statistics for different time periods and drought events.

    Parameters:
    meteorological_data (list): List of DataFrames containing meteorological variables data
    common_sites (list): List of site IDs common across all datasets
    drought_years (list): List of individual drought years
    longterm_avg_start (int): Start year for long-term average (default: 1998)
    end_year (int): End year for calculations (default: 2022)

    Returns:
    tuple: longterm precip avg, longterm temp avg, precip site data,
           temp site data, precip drought avg list, temp drought avg list
    """    
    result = {
        'site_data':       {},
        'drought_avg':     {},
        'drought_avg_list': {}
    }

    # Filter and organize site data per MET variable
    met_ready = {}
    met_allsites = {}

    for var_idx, variable in enumerate(MET_vars):
        var_data = meteorological_data[var_idx]
        common   = list(set(var_data.columns) & set(common_sites))
        result['site_data'][variable] = var_data[common]

        # Structure inputs for calculate_longterm_avg
        met_ready[variable] = var_data
        met_allsites[variable] = common

    _, longterm_basin_avgs = calculate_longterm_avg(
        met_ready, met_allsites, MET_vars, longterm_avg_start, end_year
    )

    # Drought year averages
    for var_idx, variable in enumerate(MET_vars):
        common_sites_data = result['site_data'][variable]
        drought_year_avgs = {}
        drought_avg_list = []

        for year in drought_years:
            if year in common_sites_data.index.get_level_values('wyear'):
                year_data = common_sites_data.xs(year, level='wyear')
                year_avg = year_data.mean(axis=1)
                drought_year_avgs[str(year)] = year_avg
                drought_avg_list.append(year_avg)
            else:
                print(f"Warning: Year {year} not found in {variable} data")
                drought_year_avgs[str(year)] = None

        #  Combined consecutive drought year averages 
        for i in range(len(drought_years) - 1):
            if drought_years[i] + 1 == drought_years[i + 1]:
                y1, y2 = str(drought_years[i]), str(drought_years[i + 1])
                if drought_year_avgs[y1] is not None and drought_year_avgs[y2] is not None:
                    drought_year_avgs[f'{y1}_{y2}'] = (drought_year_avgs[y1] + drought_year_avgs[y2]) / 2

        result['drought_avg'][variable]      = drought_year_avgs
        result['drought_avg_list'][variable] = drought_avg_list

    basin_avg    = {}
    congruent_sites = {}
    met_avg_list = {}

    for var_idx, variable in enumerate(MET_vars):
        basin_avg[variable]       = longterm_basin_avgs[variable]
        congruent_sites[variable] = result['site_data'][variable]
        met_avg_list[variable]    = result['drought_avg_list'][variable]

    return basin_avg, congruent_sites, met_avg_list
        

def calculate_percentile_rel_change(var_names, dr_names, refyrs_all3, dryrs_all3):
    """
    Calculate 95th and 5th percentile values and their relative changes
    for drought vs reference periods across all variables and drought episodes.

    Multi-year episodes require a groupby before quantile to get one value per year
    before taking the median. Single-year episodes can take the quantile directly.

    Parameters:
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): List of drought episode names, where index 0 and 3 are
                     multi-year episodes and index 1 and 2 are single-year episodes
    refyrs_all3 (dict): {var: {drought: DataFrame}} reference year slices
    dryrs_all3 (dict): {var: {drought: DataFrame}} drought year slices

    Returns:
    pct (dict): {var: {drought: {'ref_95', 'dr_95', 'ref_5', 'dr_5',
                                  'relchange_95', 'relchange_5'}}}
    """
    pct = {var: {dr: {} for dr in dr_names} for var in var_names}

    multi_yr_episodes  = [dr_names[0], dr_names[3]]
    single_yr_episodes = [dr_names[1], dr_names[2]]

    for var in var_names:

        # Multi-year episodes: groupby wyear then median
        for dr in multi_yr_episodes:
            pct[var][dr]['ref_95'] = refyrs_all3[var][dr].groupby(level='wyear').quantile(0.95).median()
            pct[var][dr]['dr_95']  = dryrs_all3[var][dr].groupby(level='wyear').quantile(0.95).median()
            pct[var][dr]['ref_5']  = refyrs_all3[var][dr].groupby(level='wyear').quantile(0.05).median()
            pct[var][dr]['dr_5']   = dryrs_all3[var][dr].groupby(level='wyear').quantile(0.05).median()

        # Single-year episodes: quantile directly
        for dr in single_yr_episodes:
            pct[var][dr]['ref_95'] = refyrs_all3[var][dr].groupby(level='wyear').quantile(0.95).median()
            pct[var][dr]['dr_95']  = dryrs_all3[var][dr].quantile(0.95)
            pct[var][dr]['ref_5']  = refyrs_all3[var][dr].groupby(level='wyear').quantile(0.05).median()
            pct[var][dr]['dr_5']   = dryrs_all3[var][dr].quantile(0.05)

        # Relative change for all
        for dr in dr_names:
            for pct_key, dr_key, ref_key in [('relchange_95', 'dr_95', 'ref_95'),('relchange_5',  'dr_5',  'ref_5')]:
                pct[var][dr][pct_key] = (((pct[var][dr][dr_key] - pct[var][dr][ref_key]) / pct[var][dr][ref_key]) * 100).to_frame().reset_index().rename(columns={0: 'Relative Change (%)', 'index': 'SiteId'})

    print("------ calculated relative change for peak and low RDC, WT, SC ------")
    return pct


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
    enhanced_mapping_data = {}
    
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
                var_mapping_data.at[site, 'LAT'] = float(var_metadata.loc['sampling_feature_lat', site])
                var_mapping_data.at[site, 'LON'] = float(var_metadata.loc['sampling_feature_long', site])
        
        # Add reference site classification
        var_mapping_data['CLASS'] = 'Non-Ref'
        for site in var_mapping_data.index:
            if site in reference_sites:
                var_mapping_data.at[site, 'CLASS'] = 'Ref'
        
        # For water temperature (WT): handle sites missing coordinates
        if variable == 'WT':
            # Identify sites with missing coordinates
            missing_coord_sites = var_mapping_data[
                (var_mapping_data['LAT'] == 0.0) | 
                (var_mapping_data['LON'] == 0.0)
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
        enhanced_mapping_data[variable] = var_mapping_data
    
    return enhanced_mapping_data            


def calculate_percentile_map_data(var_names, dr_names, pct, list_metadata_filtered_dfs, gagesii):
    """
    Prepare percentile relative change data for spatial mapping.
    Averages relative change across all drought episodes per variable and percentile.

    Parameters:
    var_names (list): Variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names
    pct (dict): {var: {dr: {'relchange_95': DataFrame, 'relchange_5': DataFrame}}}
                percentile relative change data per variable and drought episode
    list_metadata_filtered_dfs (list): Metadata DataFrames passed to prep_mapping
    gagesii: GAGESII data passed to prep_mapping

    Returns:
    pct_map (dict): {var: {'95': DataFrame, '5': DataFrame}} spatially-ready
                    percentile relative change data with LAT/LON/CLASS columns
    """
    pct_map = {var: {} for var in var_names}

    for val in ['95', '5']:
        relchange_ready = []

        for var in var_names:
            list_mapping_per = []

            for dr in dr_names:
                list_mapping_per.append(pct[var][dr][f'relchange_{val}'].set_index('SiteId').rename_axis(None))

            # Average relative change across all drought episodes
            mean_relchange = pd.concat(list_mapping_per).groupby(level=0).mean()
            relchange_ready.append(mean_relchange)

        mapping_return = prep_mapping(relchange_ready, list_metadata_filtered_dfs, gagesii)

        for var in var_names:
            pct_map[var][val] = mapping_return[var]

    print("------ Percentile map data prepared ------")
    return pct_map
    

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


def get_years_before_after(baselinelength, dr_years, ref_years):
    ### return dict with both
    first_last_year = []
    yearsarounddrought = []
    
    ### Get Years before drought
    first_last_year.append(dr_years[0])
    yearsarounddrought.append([x for x in ref_years if x < first_last_year[0]])
        
    ## years after drought
    first_last_year.append(dr_years[-1])
    yearsarounddrought.append([x for x in np.arange(1998,2023,1) if x > first_last_year[-1]])

    return first_last_year, yearsarounddrought


def calc_baseline(baselinelength, years_baseline, annual_medians):    
    if baselinelength == 3:
        baseline_medians = annual_medians.loc[years_baseline]
        
    if baselinelength == 2:
        baseline_medians = annual_medians.loc[years_baseline[:2]]

    ## calculate baseline for each site
    baseline_means = baseline_medians.mean(axis=0).to_frame().rename(columns={0:'baseline'})
    
    return baseline_means, baseline_medians


# Function to determine the first year meeting/exceeding the baseline
def first_year_meeting_baseline(row, relchangetype,baseline_mean):
    site = row.name  # Get the current site name from the row index
    baseline = baseline_mean.at[site, 'baseline']  # Get the baseline for that site

    if relchangetype == 'neg': # for sites with NEG relchange, we are checking when post-drought exceeds pre-drought value
        for year in row.index:
            if row[year] >= baseline:
                return year
            
    if relchangetype == 'pos': # for sites with POS relchange, we are checking when post-drought is less than pre-drought value
        for year in row.index:
            if row[year] <= baseline:
                return year
                        
    return np.nan  # Return NaN if the baseline is never met

def get_recovery_years(baselinelength, dr, dr_years, ref_years, annmedians_pos, annmedians_neg):
    first_last_year, yearsarounddrought = get_years_before_after(baselinelength, dr_years, ref_years)
    firstdryr = first_last_year[0]
    yearsbefore = yearsarounddrought[0]
    lastdryr = first_last_year[-1]
    yearsafter = yearsarounddrought[-1]
    first_year_back = []
    yearssince = []
    
    for relchangetype in ['pos','neg']:
        vars()['baseline_means_'+relchangetype], vars()['baseline_medians_'+relchangetype]  = calc_baseline(baselinelength, yearsbefore, vars()['annmedians_'+relchangetype])

        vars()['post_medians_'+relchangetype] = vars()['annmedians_'+relchangetype].loc[yearsafter]

        # Applying the function to each site (column)
        vars()['first_years_'+relchangetype] = vars()['post_medians_'+relchangetype].apply(first_year_meeting_baseline, axis=0, args=(relchangetype,vars()['baseline_means_'+relchangetype]))

        # Convert the result to DataFrame for easy readability
        vars()['firstyearback_'+relchangetype] = pd.DataFrame(vars()['first_years_'+relchangetype], columns=[dr])
        first_year_back.append(vars()['firstyearback_'+relchangetype])

        vars()['yearssince_'+relchangetype] = vars()['firstyearback_'+relchangetype] - lastdryr
        yearssince.append(vars()['yearssince_'+relchangetype])
        
    return first_year_back,yearssince


def build_annual_medians_by_rc(var_names, dr_names, MED_all3, sites_all3):
    """
    Separate sites by relative change sign and calculate annual medians
    across all drought episodes.

    Parameters:
    var_names (list): List of variable names
    dr_names (list): List of drought episode names
    MED_all3 (dict): {var: {'relchange': DataFrame}} as returned by aggregate_episodes()
    sites_all3 (dict): {var: {drought: DataFrame}} valid sites per var per episode

    Returns:
    MED_relchange_map (dict): {var: DataFrame} relative change renamed for mapping
    sites_by_rc (dict): {var: {'pos','neg','all'}} site indices by rel change sign
    combined_dr (dict): {var: DataFrame} all episodes combined via combine_first
    annual_medians (dict): {var: DataFrame} annual medians across all drought episodes
    ann_med_rc (dict): {var: {'pos','neg'}} annual medians filtered by rel change sign
    """
    MED_relchange_map = {}
    sites_by_rc       = {}
    combined_dr       = {}
    annual_medians    = {}
    ann_med_rc        = {var: {} for var in var_names}

    for var in var_names:
        # ── Rename and separate by sign ────────────────────────────────────────
        MED_relchange_map[var] = MED_all3[var]['relchange'].copy()

        col = 'Relative Change (%)'
        sites_by_rc[var] = {
            'pos': MED_relchange_map[var][MED_relchange_map[var][col] >= 0].index,
            'neg': MED_relchange_map[var][MED_relchange_map[var][col] <  0].index,
            'all': MED_relchange_map[var].index
        }

        # ── Combine all episodes and compute annual medians ────────────────────
        combined_dr[var] = (
            sites_all3[var][dr_names[0]]
            .combine_first(sites_all3[var][dr_names[1]])
            .combine_first(sites_all3[var][dr_names[2]])
            .combine_first(sites_all3[var][dr_names[3]])
        )
        annual_medians[var] = combined_dr[var].groupby(level=['wyear']).median()

        for rc in ['pos', 'neg']:
            ann_med_rc[var][rc] = annual_medians[var][sites_by_rc[var][rc]]

    print("------ separated sites by relative change sign and built annual medians ------")
    return MED_relchange_map, sites_by_rc, combined_dr, annual_medians, ann_med_rc


def calculate_recovery(var_names, dr_names, dr_years, ref_all3, ann_med_rc,
                        sites_all3, avail, numyears, firstyrback, baselinelength=3):
    """
    Calculate recovery years and data availability for each variable,
    drought episode, and relative change type.

    Parameters:
    var_names (list): List of variable names
    dr_names (list): List of drought episode names
    dr_years (dict): {drought: list} drought years per episode
    ref_all3 (dict): {drought: list} reference years per episode
    ann_med_rc (dict): {var: {'pos','neg'}} annual medians filtered by rel change sign
    sites_all3 (dict): {var: {drought: DataFrame}} valid sites per var per episode
    avail (dict): {var: DataFrame} availability data
    numyears (dict): {var: {'pos','neg'}} concatenated years since recovery
    firstyrback (dict): {var: {'pos','neg'}} concatenated first year back
    baselinelength (int): Number of baseline years (default: 3)

    Returns:
    recovery (dict): {var: {drought: {'pos','neg': {recov_years, recov_year,
                     avail_aftdr, avail_aftdr_df, notrecov, missing_aftdr,
                     notrecov_nomissing, recyears_nononrec}}}}
    dr_timing (dict): {drought: {'firstdryr','lastdryr','yearsbefore','yearsafter'}}
    """
    dr_timing = {}
    recovery  = {var: {dr: {} for dr in dr_names} for var in var_names}

    for var in var_names:
        for dr in dr_names:
            first_last_year, yearsarounddrought = get_years_before_after(
                baselinelength, dr_years[dr], ref_all3[dr]
            )

            dr_timing[dr] = {
                'firstdryr'   : first_last_year[0],
                'lastdryr'    : first_last_year[-1],
                'yearsbefore' : yearsarounddrought[0],
                'yearsafter'  : yearsarounddrought[-1]
            }
            yearsafter = dr_timing[dr]['yearsafter']

            for rc in ['pos', 'neg']:
                r = {}

                # ── Filter to sites in this episode ────────────────────────────
                sites_for_dr = list(
                    set(numyears[var][rc].index.values)
                    .intersection(set(sites_all3[var][dr].columns.values))
                )

                r['recov_years'] = numyears[var][rc].loc[sites_for_dr][dr]
                r['recov_year']  = firstyrback[var][rc].loc[sites_for_dr][dr]

                didnotrecover     = r['recov_years'].isna()
                didnotrecoverlist = r['recov_years'][didnotrecover].index

                r['avail_aftdr']    = avail[var][didnotrecoverlist].loc[yearsafter[0]:]
                r['avail_aftdr_df'] = r['avail_aftdr'].sum().to_frame().rename(columns={0: dr})
                r['missing_aftdr']  = r['avail_aftdr'].loc[
                    :, (r['avail_aftdr'] == 0).any(axis=0)
                ]

                r['notrecov'] = r['recov_years'][didnotrecover].replace(np.nan, -1)
                r['notrecov_nomissing'] = r['notrecov'].drop(index=r['missing_aftdr'].columns)
                r['notrecov_nomissing'].replace(to_replace={-1: 24}, inplace=True)

                overlapping_sites      = list(set(r['notrecov'].index).intersection(set(r['recov_years'].index)))
                r['recyears_nononrec'] = r['recov_years'].drop(index=overlapping_sites)

                recovery[var][dr][rc] = r

                # ── Print summary ──────────────────────────────────────────────
                print(var, dr, rc)
                print(f"  Did not reach recovery: {didnotrecover.sum()}"
                      f"; Missing data to 2022: {len(r['missing_aftdr'].columns)}")
                print(f"  Mean yrs:   {np.nanmean(r['recov_years']):.2f}"
                      f"; Median yrs: {np.nanmedian(r['recov_years']):.2f}"
                      f"; Max yrs:    {np.nanmax(r['recov_years']):.2f}"
                      f"; Min yrs:    {np.nanmin(r['recov_years']):.2f}")

    return recovery, dr_timing


def build_recovery_summary(var_names, dr_names, recovery):
    """
    Concatenate recovery years across drought episodes and print summary statistics.

    Parameters:
    var_names (list): List of variable names
    dr_names (list): List of drought episode names
    recovery (dict): {var: {drought: {rc: {...}}}} as returned by calculate_recovery()

    Returns:
    alldr_recyrs_df (dict): {var: {'pos','neg': DataFrame}} recovery years as DataFrame
    """
    alldr_recyrs_df = {var: {} for var in var_names}

    for var in var_names:
        for rc in ['neg', 'pos']:
            alldr_recyrs_df[var][rc] = pd.concat(
                {dr: recovery[var][dr][rc]['recyears_nononrec'] for dr in dr_names[:-1]},
                axis=1
            )

            print(f"\n{var} | {rc}")
            for dr in dr_names[:-1]:
                s = recovery[var][dr][rc]['recyears_nononrec']
                print(f"  {dr}: # sites recovered: {len(s.index)}, median yrs: {s.median():.2f}")

            overall_median = alldr_recyrs_df[var][rc].median(axis=1).median()
            print(f"  All episodes — # sites: {len(alldr_recyrs_df[var][rc].index)}, "
                  f"median yrs: {overall_median:.2f}")

    print('\nSites that did not recover with positive or negative rel change during drought:')
    for var in var_names:
        for dr in dr_names[:-1]:
            print(var, dr)
            for rc in ['neg', 'pos']:
                print(rc, np.sort(recovery[var][dr][rc]['notrecov_nomissing'].index.values))

    return alldr_recyrs_df


def build_recovery_year_dicts(var_names, dr_names, dr_years, ref_all3,
                               ann_med_rc, baselinelength=3):
    """
    Accumulate first year back and years since recovery across all drought episodes.

    Parameters:
    var_names (list): List of variable names
    dr_names (list): List of drought episode names
    dr_years (dict): {drought: list} drought years per episode
    ref_all3 (dict): {drought: list} reference years per episode
    ann_med_rc (dict): {var: {'pos','neg'}} annual medians by rel change sign
    baselinelength (int): Number of baseline years (default: 3)

    Returns:
    firstyrback (dict): {var: {'pos','neg'}} concatenated first year back across episodes
    numyears (dict): {var: {'pos','neg'}} concatenated years since recovery across episodes
    """
    alldrs_rc     = {var: {'pos': [], 'neg': []} for var in var_names}
    yearssince_rc = {var: {'pos': [], 'neg': []} for var in var_names}
    firstyearback = {var: {dr: {} for dr in dr_names} for var in var_names}
    yrs_since     = {var: {dr: {} for dr in dr_names} for var in var_names}

    for var in var_names:
        for dr in dr_names:
            first_year_back, yrs_since_dr = get_recovery_years(
                baselinelength, dr, dr_years[dr], ref_all3[dr],
                ann_med_rc[var]['pos'], ann_med_rc[var]['neg']
            )

            firstyearback[var][dr]['pos'] = first_year_back[0]
            firstyearback[var][dr]['neg'] = first_year_back[-1]
            yrs_since[var][dr]['pos']     = yrs_since_dr[0]
            yrs_since[var][dr]['neg']     = yrs_since_dr[-1]

            for rc in ['pos', 'neg']:
                alldrs_rc[var][rc].append(firstyearback[var][dr][rc])
                yearssince_rc[var][rc].append(yrs_since[var][dr][rc])

    firstyrback = {
        var: {rc: pd.concat(alldrs_rc[var][rc], axis=1) for rc in ['pos', 'neg']}
        for var in var_names
    }
    numyears = {
        var: {rc: pd.concat(yearssince_rc[var][rc], axis=1) for rc in ['pos', 'neg']}
        for var in var_names
    }

    print("------ calculated number of years to pre-event baseline ------")
    return firstyrback, numyears

    

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

        newdf.at[site, 'startyr']    = int(startyr)
        newdf.at[site, 'endyr']      = int(endyr)
        newdf.at[site, 'totalyrs']   = int(totalyrs)
        newdf.at[site, 'trend']      = str(trend)
        newdf.at[site, 'h']          = bool(h)
        newdf.at[site, 'p']          = float(p)
        newdf.at[site, 'z']          = float(z)
        newdf.at[site, 'tau']        = float(tau)
        newdf.at[site, 's']          = float(s)
        newdf.at[site, 'var_s']      = float(var_s)
        newdf.at[site, 'slope']      = float(slope)
        newdf.at[site, 'intercept']  = float(intercept)
    return newdf

def run_mann_kendall_all(ready, met_wyear, runoff_efficiency_ann, reservoirs_years,
                          var_names, MET_vars, allsites, start_year=1998, end_year=2022):
    """
    Filter data to sites of interest, average to water years, and run
    Mann-Kendall test for all variables.

    Parameters:
    ready (dict): {var: DataFrame} normalized data keyed by variable name
    met_wyear (dict): {MET: DataFrame} water year MET data
    runoff_efficiency_ann (DataFrame): Annual runoff efficiency data
    reservoirs_years (DataFrame): Annual reservoir data prepped by prep_reservoir_data()
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    MET_vars (list): List of MET variable names e.g. ['precip', 'temp']
    allsites (dict): {var: list} union of valid sites per variable
    start_year (int): Start year for filtering (default: 1998)
    end_year (int): End year for filtering (default: 2022)

    Returns:
    mk (dict): {var: {'template': DataFrame, 'results': DataFrame}}
    years (dict): {var: DataFrame} annual water year data per variable
    """
    columnvals    = ['startyr', 'endyr', 'totalyrs', 'trend', 'h', 'p', 'z',
                     'tau', 's', 'var_s', 'slope', 'intercept']

    # Step 1: Filter sites
    withsites = {}
    for var in var_names:
        withsites[var] = ready[var][allsites[var]]
    for MET in MET_vars:
        withsites[MET] = met_wyear[MET][allsites['RDC']]

    # Step 2 & 3: Mean to water years and isolate period
    ann   = {}
    years = {}

    for var in all_variables:
        ann[var]   = withsites[var].groupby(level=[0]).mean()
        years[var] = ann[var].loc[start_year:end_year]
        years[var] = years[var].dropna(how='all', axis=1)

    years['runoff_efficiency'] = runoff_efficiency_ann[allsites['RDC']]
    years['reservoirs']        = reservoirs_years

    # Step 4: Mann-Kendall test
    mk = {}
    for var in all_vars_qp:
        mk[var] = {}
        mk[var]['template'] = pd.DataFrame(index=years[var].columns, columns=columnvals)
        mk[var]['results']  = mann_kendall_table(years[var], mk[var]['template'])

    print("------ computed MK test for RDC, WT, SC, precip, temp, runoff_efficiency, reservoirs ------")
    return mk, years, withsites


def filter_mk_trends(mk):
    """
    Filter Mann-Kendall results into no trend, increasing, and decreasing subsets.

    Parameters:
    mk (dict): {var: {'results': DataFrame}} as returned by run_mann_kendall_all()
    all_vars_qp (list): List of all variable names including runoff_efficiency and reservoirs

    Returns:
    mk_trends (dict): {var: {'notrend', 'increasing', 'decreasing'}}
    """
    mk_trends = {var: {} for var in all_vars_qp}

    for var in all_vars_qp:
        print(var, ', Sites per each type of trend result:')
        print(mk[var]['results']['trend'].value_counts())

        mk_trends[var]['notrend']    = mk[var]['results'][mk[var]['results']['trend'] == 'no trend']
        mk_trends[var]['increasing'] = mk[var]['results'][mk[var]['results']['trend'] == 'increasing']
        mk_trends[var]['decreasing'] = mk[var]['results'][mk[var]['results']['trend'] == 'decreasing']

    return mk_trends

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

        # Check for positive values (Box-Cox requires strictly positive) << not really an issue since all are positive
        min_val = valid.min()
        if min_val <= 0:
            # Shift data by abs(min_val) + 1% of mean to normalize
            mean_val = valid[valid > 0].mean()
            epsilon = mean_val * 0.01  # 1% of the mean
            shift = abs(min_val) + epsilon
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

def run_regression_analysis(runoff_years, P_years, list_dr_years):
    """
    Apply Box-Cox transformation to runoff, run drought regressions,
    and filter results by significance and direction.

    Parameters:
    runoff_years (DataFrame): Annual runoff data as returned by prep_regression_data()
    P_years (DataFrame): Annual precipitation data as returned by prep_regression_data()
    list_dr_years (list): List of lists of drought years per episode
                          e.g. [[2001,2002],[2012],[2018],[2020,2021]]

    Returns:
    results (DataFrame): Regression results with columns a0, a1, a2, pval_a1, rho
    sig_pos (array): Site IDs with significant positive a1
    sig_neg (array): Site IDs with significant negative a1
    notsig (array): Site IDs with non-significant a1
    runoff_boxcox (DataFrame): Box-Cox transformed runoff data
    I_all: df signifying drought and nondrought
    """
    # Flatten drought years across all episodes
    dryears = [yr for episode in list_dr_years for yr in episode]

    # Box-Cox transformation
    runoff_boxcox, lambdas = boxcox_transform_dataframe(runoff_years)

    # Run regressions
    results, I_all = run_drought_regressions(runoff_boxcox, P_years, dryears)

    # Filter by significance and direction
    sig_pos = results[(results['pval_a1'] < 0.05) & (results['a1'] >= 0)].index.values
    sig_neg = results[(results['pval_a1'] < 0.05) & (results['a1'] <  0)].index.values
    notsig  = results[results['pval_a1'] >= 0.05].index.values

    print(f'Significant a1:          {len(results[results["pval_a1"] < 0.05])}')
    print(f'Positive significant a1: {len(sig_pos)}')
    print(f'Negative significant a1: {len(sig_neg)}')
    print(f'Not significant a1:      {len(notsig)}')

    return results, sig_pos, sig_neg, notsig, runoff_boxcox, I_all

def calculate_prepost_rel_change(var_names, MET_vars, dr_names, dr_years, ref_all3,
                                  annual_medians, sites_all3, mk_trends,
                                  all_vars_qp, mk_types, start_year=1998, end_year=2022):
    """
    Calculate pre-to-post drought relative change and filter by Mann-Kendall trend type.

    Parameters:
    var_names (list): List of variable names e.g. ['RDC', 'WT', 'SC']
    MET_vars (list): List of MET variable names e.g. ['precip', 'temp']
    dr_names (list): List of drought episode names
    dr_years (dict): {drought: list} drought years per episode
    ref_all3 (dict): {drought: list} reference years per episode
    annual_medians (dict): {var: DataFrame} annual medians per variable
    sites_all3 (dict): {var: {drought: DataFrame}} valid sites per var per episode
    mk_trends (dict): {var: {'notrend','increasing','decreasing'}} MK filtered results
    all_vars_qp (list): All variable names including runoff_efficiency and reservoirs
    mk_types (list): List of MK trend types e.g. ['notrend','increasing','decreasing']
    start_year (int): Start year for filtering (default: 1998)
    end_year (int): End year for filtering (default: 2022)

    Returns:
    yearsbefore (dict): {drought: list} reference years before drought
    yearsafter (dict): {drought: list} reference years after drought
    prepost_rc (dict): {var: {drought: DataFrame}} pre-to-post relative change
    prepost_rc_mk (dict): {var: {drought: {mk_type: DataFrame}}} relchange by MK trend
    annual_medians (dict): filtered to study period
    """
    
    #  Filter annual medians to study period 
    for var in all_vars_qp:
        annual_medians[var] = annual_medians[var].loc[start_year:end_year]
        annual_medians[var] = annual_medians[var].dropna(how='all', axis=0)

    # Split reference years into before/after drought
    yearsbefore = {}
    yearsafter  = {}

    for dr in dr_names:
        dryr1            = dr_years[dr][0]
        dryr2            = dr_years[dr][-1]
        yearsbefore[dr]  = [x for x in ref_all3[dr] if x < dryr1]
        yearsafter[dr]   = [x for x in ref_all3[dr] if x > dryr2]

    # Calculate pre-to-post relative change and filter by MK trend 
    prepost_rc    = {var: {} for var in all_vars_qp}
    prepost_rc_mk = {var: {dr: {} for dr in dr_names} for var in all_vars_qp}

    for var in all_vars_qp:
        print(var)
        for dr in dr_names:
            # Select correct sites depending on variable type
            if var in var_names:
                src = annual_medians[var][sites_all3[var][dr].columns.values]
            
            if var == 'runoff_efficiency':
                src = annual_medians[var][sites_all3['RDC'][dr].columns.values]
            else:
                src = annual_medians[var]

            pre  = src.loc[yearsbefore[dr]].median()
            post = src.loc[yearsafter[dr]].median()
            ref  = pre.replace(0, 1e-10)

            prepost_rc[var][dr] = ((post - ref) / ref.abs() * 100).to_frame()

            print(f"  {dr}")
            for mk_type in mk_types:
                shared_sites = prepost_rc[var][dr].index.intersection(
                    mk_trends[var][mk_type].index
                )
                prepost_rc_mk[var][dr][mk_type] = prepost_rc[var][dr].loc[shared_sites]
                n      = len(shared_sites)
                median = (prepost_rc_mk[var][dr][mk_type].median().values[0]
                          if n > 0 else float('nan'))
                print(f"    {mk_type:12s} — sites: {n}, "
                      f"Rel. Change Post from Pre drought: {median:.4f}")

    return yearsbefore, yearsafter, prepost_rc, prepost_rc_mk, annual_medians

    
def calculate_ref_period_rel_change(all_vars_qp, dr_names, ann_MED, met_ann_MED,
                                           annual_medians, ref_all3):
    """
    Calculate relative change between reference periods of different drought episodes,
    using the first drought episode as the baseline.

    Parameters:
    all_vars_qp (list): All variable names including runoff_efficiency and reservoirs
    dr_names (list): List of drought episode names, where dr_names[0] is the baseline
    ann_MED (dict): {var: {drought: {'ref': DataFrame}}} annual median results for RDC/WT/SC
    met_ann_MED (dict): {MET: {drought: {'ref': DataFrame}}} annual median results for MET vars
    annual_medians (dict): {var: DataFrame} annual medians per variable
    ref_all3 (dict): {drought: list} reference years per episode

    Returns:
    refMED_rc (dict): {var: {(dr_base, dr_comp): Series}} relative change between
                      reference periods keyed by episode pair tuple
    """
    refMED_rc = {var: {} for var in all_vars_qp}
    dr_base   = dr_names[0]

    for var in all_vars_qp:
        print(var)

        for dr_comp in dr_names[1:]:

            if var in ['RDC', 'WT', 'SC']:
                shared_sites = ann_MED[var][dr_base]['ref'].index.intersection(
                               ann_MED[var][dr_comp]['ref'].index)
                print(f"  # sites in common {dr_base} and {dr_comp}: {len(shared_sites)}")

                base = ann_MED[var][dr_base]['ref'].loc[shared_sites].replace(0, 1e-10)['REF']
                comp = ann_MED[var][dr_comp]['ref'].loc[shared_sites]['REF']

            elif var in ['precip', 'temp']:
                base = met_ann_MED[var][dr_base]['ref'].replace(0, 1e-10)
                comp = met_ann_MED[var][dr_comp]['ref']

            elif var == 'runoff_efficiency':
                RDC_sites = ann_MED['RDC'][dr_base]['ref'].index.intersection(ann_MED['RDC'][dr_comp]['ref'].index)
                base = annual_medians[var].loc[ref_all3[dr_base]].median().replace(0, 1e-10)[RDC_sites].dropna(how='all')
                comp = annual_medians[var].loc[ref_all3[dr_comp]].median()[RDC_sites].dropna(how='all')
                shared_sites = base.index.intersection(comp.index)
                print(f"  # sites in common {dr_base} and {dr_comp}: {len(shared_sites)}")
                base = base[shared_sites]
                comp = comp[shared_sites]
            
            elif var in ['reservoirs']:
                base = annual_medians[var].loc[ref_all3[dr_base]].median().replace(0, 1e-10).dropna(how='all')
                comp = annual_medians[var].loc[ref_all3[dr_comp]].median().dropna(how='all')
                shared_sites = base.index.intersection(comp.index)
                print(f"  # sites in common {dr_base} and {dr_comp}: {len(shared_sites)}")
                base = base[shared_sites]
                comp = comp[shared_sites]

            refMED_rc[var][(dr_base, dr_comp)] = ((comp - base) / base) * 100
            print(f"  Rel. change of {dr_comp} ref. from {dr_base} ref.: "
                  f"{np.median(refMED_rc[var][(dr_base, dr_comp)]):.4f}")

    return refMED_rc


def calculate_pre_period_rel_change(all_vars_qp, dr_names, ann_MED, annual_medians, yearsbefore):
    """
    Calculate relative change between pre-drought periods of different drought episodes,
    using the first drought episode as the baseline.

    Parameters:
    all_vars_qp (list): All variable names including runoff_efficiency and reservoirs
    dr_names (list): List of drought episode names, where dr_names[0] is the baseline
    ann_MED (dict): {var: {drought: {'ref': DataFrame}}} used to get shared sites for RDC/WT/SC
    annual_medians (dict): {var: DataFrame} annual medians per variable
    yearsbefore (dict): {drought: list} pre-drought years per episode

    Returns:
    preMED_rc (dict): {var: {(dr_base, dr_comp): Series}} relative change between
                      pre-drought periods keyed by episode pair tuple
    """
    preMED_rc = {var: {} for var in all_vars_qp}
    dr_base   = dr_names[0]

    for var in all_vars_qp:
        print(var)

        for dr_comp in dr_names[1:]:

            if var in ['RDC', 'WT', 'SC']:
                shared_sites = ann_MED[var][dr_base]['ref'].index.intersection( ann_MED[var][dr_comp]['ref'].index)

                base = annual_medians[var].loc[yearsbefore[dr_base], shared_sites].median().replace(0, 1e-10)
                comp = annual_medians[var].loc[yearsbefore[dr_comp], shared_sites].median()

            elif var in ['precip', 'temp']:
                base = annual_medians[var].loc[yearsbefore[dr_base]].median().replace(0, 1e-10)
                comp = annual_medians[var].loc[yearsbefore[dr_comp]].median()

            elif var == 'runoff_efficiency':
                RDC_sites = ann_MED['RDC'][dr_base]['ref'].index.intersection(ann_MED['RDC'][dr_comp]['ref'].index)

                base = annual_medians[var].loc[yearsbefore[dr_base]].median().replace(0, 1e-10)[RDC_sites].dropna(how='all')
                comp = annual_medians[var].loc[yearsbefore[dr_comp]].median()[RDC_sites].dropna(how='all')
                shared_sites = base.index.intersection(comp.index)
                print(f"  # sites in common {dr_base} and {dr_comp}: {len(shared_sites)}")
                base = base[shared_sites]
                comp = comp[shared_sites]
            
            elif var == 'reservoirs':
                base = annual_medians[var].loc[yearsbefore[dr_base]].median().replace(0, 1e-10).dropna(how='all')
                comp = annual_medians[var].loc[yearsbefore[dr_comp]].median().dropna(how='all')

                shared_sites = base.index.intersection(comp.index)
                base = base[shared_sites]
                comp = comp[shared_sites]

            preMED_rc[var][(dr_base, dr_comp)] = ((comp - base) / base) * 100
            print(f"  Rel. change of {dr_comp} pre from {dr_base} pre: "
                  f"{np.median(preMED_rc[var][(dr_base, dr_comp)]):.4f}")

    return preMED_rc


def get_var_onemonth_timeseries(var_all, month):
    idx = pd.IndexSlice
    var_month = var_all.groupby(level=['wyear', 'month']).median().loc[idx[:, month], :].groupby(level=['wyear']).median()
    sorted_var_list = var_month.mean().sort_values(ascending=False).index.values
    sorted_var_month = var_month[sorted_var_list]

    # get the annual medians for WT sites to be plotted - SORTED BY the MONTH VALUES!!!!
    sorted_var_ann = var_all.groupby(level=['wyear']).median()[sorted_var_list]
    return (sorted_var_ann,sorted_var_month)


def calc_values_RE(dr_names,ref_all3, dr_all3, dr_years, data, site_2_use, available):

    ann_MED_RE = {var: {} for var in ['RE']}

    for dr in dr_names:
        ann_MED_RE['RE'][dr] = {}
        df_ref = data[data.index.get_level_values('wyear').isin(ref_all3[dr])][site_2_use[dr].columns]
        df_dr  = data[data.index.get_level_values('wyear').isin(dr_years[dr])][site_2_use[dr].columns]
        (ann_MED_RE['RE'][dr]['relchange'],ann_MED_RE['RE'][dr]['drought'],ann_MED_RE['RE'][dr]['ref'],ann_MED_RE['RE'][dr]['diff'] ) = rel_change_median_annual(df_ref, df_dr, dr)

    # Sites with data for each drought episode 
    sites_all3_RE = {var: {} for var in ['RE']}

    for num, var in enumerate(['RE']):
        for drought in dr_names:
            sites_all3_RE[var][drought] = data.copy()

            for site in available.columns:
                for year in dr_all3[drought]:
                    if available.at[year, site] == 0:
                        if site in sites_all3_RE['RE'][drought].columns.values:
                            sites_all3_RE['RE'][drought].drop(site, axis=1, inplace=True)
    avail_RE = {}
    avail_RE['RE'] = available

    return ann_MED_RE, sites_all3_RE, avail_RE