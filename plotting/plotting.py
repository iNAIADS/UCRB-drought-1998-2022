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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader as shpreader
from cartopy.feature import ShapelyFeature
from matplotlib.colorbar import ColorbarBase


var_names = ['RDC','WT','SC']
MET_vars = ['precip','temp']

def setup_plotting(dirs, dr_names=None):
    """
    Prepare figure output directories and load all shared plotting settings.

    Parameters:
    dirs (dict): Directory dictionary containing 'outputs_dir' key
    dr_names (list): Drought episode names in order, used to build per-episode style dicts.
                     Defaults to ['2001_2002', '2012', '2018', '2020_2021'] if None.

    Returns:
    dict with keys:
        'fig_dir'             (str)   : Path to figures output directory
        'data_for_figures'    (Path)  : Path to FIGURE_data directory
        'spei_cmap'           (list)  : SPEI diverging palette
        'episode_cmap'        (list)  : 4-colour colorblind palette for episodes
        'met_cmap'            (list)  : 5-colour colorblind palette for met variables
        'dark_bwr'            (list)  : color palette for Q Rel Change
        'dark_puor'           (list)  : color palette for WT and SC Rel Change
        'episode_markers'     (list)  : Marker styles in episode order
        'episode_linestyles'  (list)  : Line styles in episode order
        'dr_markerstyles'     (dict)  : {dr: marker}
        'dr_linestyles'       (dict)  : {dr: linestyle}
        'dr_colors'           (dict)  : {dr: colour}
        'wmon_names'          (list)  : Water year month name labels
    """
    if dr_names is None:
        dr_names = ['2001_2002', '2012', '2018', '2020_2021']

    fig_dir          = os.path.join(dirs['outputs_dir'], 'Figures/')
    os.makedirs(fig_dir, exist_ok=True)

    data_for_figures          = os.path.join(dirs['outputs_dir'], 'FIGURE_data/')
    os.makedirs(data_for_figures, exist_ok=True)

    spei_cmap    = sns.diverging_palette(20, 250, n=6, center="light", as_cmap=False)
    episode_cmap = sns.color_palette("colorblind", n_colors=4)
    met_cmap     = sns.color_palette("colorblind", n_colors=5)

    dark_bwr_colors = [
    "#49000a", "#67000d", "#a50026", "#cb181d", "#ef3b2c",
    "#f7f7f7",
    "#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b",
    ]
    dark_bwr = mcolors.LinearSegmentedColormap.from_list("dark_bwr", dark_bwr_colors)

    dark_puor_colors = [
    "#1a0050", "#2d0073", "#4a0099", "#6a3db5", "#b8a9d9",
    "#f5f5f5",
    "#fdd9b0", "#f59434", "#c45e00", "#8b3a00", "#5c2000",
    ]
    dark_puor = mcolors.LinearSegmentedColormap.from_list("dark_puor", dark_puor_colors)


    # Per-episode styles
    marker_cycle    = ['o', '*', 's', 'P']
    linestyle_cycle = ['solid', 'dashed', 'dashdot', 'dotted']

    episode_markers    = marker_cycle[:len(dr_names)]
    episode_linestyles = linestyle_cycle[:len(dr_names)]

    dr_markerstyles = {dr: marker_cycle[i]    for i, dr in enumerate(dr_names)}
    dr_linestyles   = {dr: linestyle_cycle[i] for i, dr in enumerate(dr_names)}
    dr_colors       = {dr: episode_cmap[i]    for i, dr in enumerate(dr_names)}

    # Month labels (water year order)
    wmon_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar',
                  'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

    # Global font settings (Arial per editorial guidelines)
    rcParams['font.family']       = 'sans-serif'
    rcParams['font.sans-serif']   = ['Arial']
    rcParams['mathtext.fontset']  = 'stixsans'

    print("------ Plotting environment configured ------")

    return {
        'fig_dir':            fig_dir,
        'data_for_figures':   data_for_figures,
        'spei_cmap':          spei_cmap,
        'episode_cmap':       episode_cmap,
        'met_cmap':           met_cmap,
        'dark_bwr':           dark_bwr,
        'dark_puor':          dark_puor,
        'episode_markers':    episode_markers,
        'episode_linestyles': episode_linestyles,
        'dr_markerstyles':    dr_markerstyles,
        'dr_linestyles':      dr_linestyles,
        'dr_colors':          dr_colors,
        'wmon_names':         wmon_names,
    }
    

def plot_data_availability(avail_plot, n, episode_cmap, fig_dir=None, save=False):
    """
    Plot annual data availability heatmaps for RDC, WT, and SC variables.

    Parameters:
    avail_plot (dict): {var: DataFrame} binary availability matrices for plotting
    n (dict): {var: int} site counts per variable
    episode_cmap (list): list of colours, indices 0/1/2 used for RDC/WT/SC respectively
    fig_dir (str): directory to save figures (required if save=True)
    save (bool): whether to save the figure to disk

    Returns:
    fig, axes
    """
    var_labels = {
        'RDC': ('a) Q Data Availability',  0),
        'WT':  ('b) WT Data Availability', 1),
        'SC':  ('c) SC Data Availability', 2),
    }

    fig, axes = plt.subplots(3, 1, figsize=(15, 20))

    for ax, (var, (title_prefix, cmap_idx)) in zip(axes, var_labels.items()):
        sns.heatmap(
            avail_plot[var],
            cmap=mpl.colors.ListedColormap(['white', episode_cmap[cmap_idx]]),
            ax=ax,
            cbar=False
        )
        ax.set_title(f'{title_prefix}, n: {n[var]}', fontsize=16)
        ax.set_ylabel('')

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'dataavailability.svg'),
                    format='svg', transparent=True, dpi=300)
        fig.savefig(os.path.join(fig_dir, 'dataavailability.jpeg'), dpi=300)
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, axes



# plotting SPEI
def plotting_spei(ann_spei_wy, gagesii_int, RDC_all3_allsites, spei_cmap, figure_dir=None, OPTION=False):
    '''
    Plot the SPEI for given sites in elevation order and get table of SPEI values
    '''

    if not isinstance(spei_cmap, mcolors.Colormap):
        mpl_cmap = mcolors.LinearSegmentedColormap.from_list('spei_cmap', spei_cmap)
    else:
        mpl_cmap = spei_cmap

    boundaries = [-1.5, -1.0, -0.5,0, 0.5, 1.0, 1.5] # for colorbar at bottom
    n_classes = len(boundaries) - 1  # 5 intervals between 6 boundaries
    class_colors = [mpl_cmap(i / (n_classes - 1)) for i in range(n_classes)]
    discrete_cmap = mcolors.ListedColormap(class_colors)
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    ##### Add columns to make basin avg wider
    ann_spei = ann_spei_wy.copy()
    ann_spei['Basin Average'] = ann_spei['BASIN_AVG']
    ann_spei['BASIN_AVG'] = np.nan
    ann_spei['Basin Avg'] = ann_spei['Basin Average']
    ann_spei['BASIN Avg'] = ann_spei['Basin Average']
    ann_spei['Basin AVG'] = ann_spei['Basin Average']

    #### Sort sites by DESCENDING MEAN ELEVATION
    gages_elevation = gagesii_int[['ELEV_MEAN_M_BASIN']]
    total_size_all3 = len(RDC_all3_allsites)
    size_plus5_all3 = total_size_all3 + 5
    gages_sort_elev_all3 = gages_elevation.transpose()[RDC_all3_allsites].sort_values(
        by='ELEV_MEAN_M_BASIN', axis=1, ascending=False
    )
    sorted_sites_all3 = np.concatenate(
        (gages_sort_elev_all3.columns.values.reshape((1, total_size_all3)),
         ann_spei.columns[-5:].values.reshape((1, 5))), axis=1
    )
    sorted_sites_2_all3 = sorted_sites_all3.reshape((size_plus5_all3,))
    sorted_sites_2_all3 = list(map(str, sorted_sites_2_all3))
    ann_spei_2plot = ann_spei[sorted_sites_2_all3]
    row_means = ann_spei_2plot['Basin Average']
    drought_years = row_means[row_means <= -1].index.tolist()

    # Elevation stats
    elev_vals   = gages_sort_elev_all3.loc['ELEV_MEAN_M_BASIN']
    max_elev    = elev_vals.max()
    min_elev    = elev_vals.min()
    median_elev = elev_vals.median()
    mean_elev   = elev_vals.mean()
    max_site    = elev_vals.idxmax()
    min_site    = elev_vals.idxmin()

    fig = plt.figure(figsize=(20, 11))
    ax_heat = fig.add_axes([0.05, 0.30, 0.90, 0.65])   # heatmap
    ax_cbar = fig.add_axes([0.15, 0.13, 0.70, 0.04])   # colorbar

    # --- Heatmap ---
    sns.heatmap(ann_spei_2plot,annot=False,fmt=".1f",
        cmap=discrete_cmap,   # use discrete colormap
        norm=norm,            # use BoundaryNorm
        ax=ax_heat,cbar=False,linewidths=0,linecolor='none')

    # Y-axis labels
    ax_heat.set_ylabel('')
    ax_heat.yaxis.set_tick_params(length=0)
    for label in ax_heat.get_yticklabels():
        label.set_fontsize(13)   # (2) bigger font

    # X-axis: hide
    ax_heat.set_xticklabels([])
    ax_heat.xaxis.set_tick_params(length=0)
    ax_heat.set_xlabel('')

    # Site count — upper right
    ax_heat.text(1.0, 1.015,f'n: {len(RDC_all3_allsites)} sites',transform=ax_heat.transAxes,
        fontsize=13, ha='right', va='bottom', color='black'   # (2) bigger font
    )

    # Right-side drought year labels
    years = list(ann_spei_2plot.index)
    for yr in drought_years:
        if yr in years:
            y_pos = years.index(yr) + 0.5
            ax_heat.text(1.002, y_pos,str(yr),transform=ax_heat.get_yaxis_transform(),
                fontsize=11, ha='left', va='center',          # (2) bigger font
                color='#b22222', fontweight='bold'
            )

    # Elevation annotation text in figure coordinates
    header_y = 0.285
    stat1_y  = 0.255
    stat2_y  = 0.228

    # Left — Higher Elevation
    fig.text(0.05, header_y, 'Higher Elevation Sites',fontsize=13, ha='left', va='top', fontweight='bold', color='black')   # (2)
    fig.text(0.05, stat1_y,  f'Max: {max_elev:.1f} m',fontsize=12, ha='left', va='top', color='black')
    fig.text(0.05, stat2_y,  f'SiteID: {max_site}',fontsize=12, ha='left', va='top', color='black')

    # Center — Median
    fig.text(0.50, header_y, f'Median: {median_elev:.2f} m',
             fontsize=13, ha='center', va='top', color='black')

    # Right — Lower Elevation
    fig.text(0.78, header_y, 'Lower Elevation Sites',fontsize=13, ha='left', va='top', fontweight='bold', color='black')
    fig.text(0.78, stat1_y,  f'Min: {min_elev:.1f} m',fontsize=12, ha='left', va='top', color='black')
    fig.text(0.78, stat2_y,  f'SiteID: {min_site}',fontsize=12, ha='left', va='top', color='black')

    # Far right — Basin Avg
    fig.text(0.91, header_y, 'Basin Avg',fontsize=13, ha='left', va='top', fontweight='bold', color='black')
    fig.text(0.91, stat1_y,  f'Mean: {mean_elev:.1f} m',fontsize=12, ha='left', va='top', color='black')

    #  Discrete colorbar with tick labels and "SPEI" title
    cb = ColorbarBase(ax_cbar,cmap=discrete_cmap, norm=norm,orientation='horizontal',
        ticks=boundaries,          # tick at each boundary edge
        spacing='uniform'
    )
    cb.ax.set_xticklabels(['-1.5', '-1.0', '-0.5','0', '0.5', '1.0', '1.5'], fontsize=12)  # (2)
    cb.outline.set_visible(False)

    # (3) "SPEI" as the colorbar title (appears above the bar)
    cb.ax.set_title('SPEI', fontsize=17, fontweight='bold', pad=6)

    # Labels below the colorbar
    label_y1 = 0.095   # Dry / Normal / Wet
    label_y2 = 0.068   # DROUGHT
    fig.text(0.225, label_y1, 'Dry',     fontsize=13, ha='center', va='top', color='black')  # (2)
    fig.text(0.50,  label_y1, 'Normal',  fontsize=13, ha='center', va='top', color='black')
    fig.text(0.775, label_y1, 'Wet',     fontsize=13, ha='center', va='top', color='black')
    fig.text(0.225, label_y2, 'DROUGHT', fontsize=14, ha='center', va='top',color='black', fontweight='bold')

    if figure_dir is not None:
        plt.savefig(figure_dir + '/ALL3_SPEI_wy_elevation_VARSITES_wlabels.svg', dpi=300, bbox_inches='tight')
        plt.savefig(figure_dir + '/ALL3_SPEI_wy_elevation_VARSITES_wlabels.jpeg', dpi=300, bbox_inches='tight')
    
    plt.show()

    if OPTION:
        gages_sort_elev_all3['Basin Average'] = gages_sort_elev_all3.loc['ELEV_MEAN_M_BASIN'].mean()
        spei_2save = ann_spei_2plot.copy().transpose()
        spei_2save['Elevation'] = gages_sort_elev_all3.transpose()['ELEV_MEAN_M_BASIN']
        spei_2save.to_csv(figure_dir + 'spei_ann_wyElevation.csv')
        print('SPEI and elevation dataframe saved: ' + figure_dir + 'spei_ann_wyElevation.csv')
    return

    

def plot_dr_ref_scatter(var_names, dr_names, ann_MED_drought, ann_MED_ref,
                                   episode_cmap, episode_markers, episode_linestyles,
                                   var_plot_names, fig_dir=None, save=False):
    """
    Scatterplot of annual medians: drought vs reference period per episode.

    Parameters:
    var_names (list): Variable names in plot order e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names
    ann_MED_drought (dict): {var: {dr: Series}} drought period annual medians
    ann_MED_ref (dict): {var: {dr: Series}} reference period annual medians
    episode_cmap (list): Colours per drought episode
    episode_markers (list): Marker styles per drought episode
    episode_linestyles (list): Line styles per drought episode
    var_plot_names (dict): {var: (axis_label, panel_letter, ylim, xlim, axline_pt)}
                           where axline_pt is a tuple (x, y) for the 1:1 line anchor
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save the figure to disk

    Returns:
    fig, axes
    """
    fsize     = 18
    fontsize2 = fsize - 2

    fig, axes = plt.subplots(nrows=len(var_names), ncols=1, figsize=(9, 16))

    for i, ax in enumerate(axes):
        var                                          = var_names[i]
        plot_name, letter, ylim, xlim, axline_pt    = var_plot_names[var]

        ax.axline(xy1=axline_pt, xy2=None, slope=1, color='gray', linestyle='-')
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_xlim(left=xlim[0],  right=xlim[1])
        ax.set_ylabel('Drought '   + plot_name, fontsize=fontsize2)
        ax.set_xlabel('Reference ' + plot_name, fontsize=fontsize2)
        ax.tick_params(axis='y')

        for num, dr in enumerate(dr_names):
            var_drought   = ann_MED_drought[var][dr]
            var_reference = ann_MED_ref[var][dr]

            ax.scatter(
                x=var_reference, y=var_drought,
                color=episode_cmap[num], s=30,
                alpha=0.5, marker=episode_markers[num]
            )

            z = np.polyfit(x=var_reference, y=var_drought, deg=1)
            a = z[0]
            p = np.poly1d(z)
            ax.plot(
                var_reference, p(var_reference),
                color=episode_cmap[num],
                linestyle=episode_linestyles[num],
                linewidth=2,
                label=f"{dr}, m= {a:.2f}, n: {len(var_drought)}"
            )

        for label in ax.get_yticklabels():
            label.set_fontsize(fontsize2)
        for label in ax.get_xticklabels():
            label.set_fontsize(fontsize2)

        ax.set_title(f'{letter} {plot_name}', fontsize=fsize)
        ax.legend(loc='best', prop={"size": fontsize2})

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'ALL3_dr_ep_scatter_GAP.jpeg'), dpi=300)
        fig.savefig(os.path.join(fig_dir, 'ALL3_dr_ep_scatter_GAP.svg'),
                    format='svg', transparent=True, dpi=300)
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, axes


def plot_combined_monthly_analysis(var_names, dr_names, MET_vars,
                                    mon_MED_drought, mon_MED_relchange, basin_avg,
                                    wmon_names, dr_colors, dr_markerstyles, dr_linestyles,
                                    var_plot_names, var_rel_names, met_plot_names,
                                    longterm_avg_start=1998, with_precip_temp=True,
                                    fig_dir=None, save=False):
    """
    Plot combined seasonal analysis: absolute monthly medians and relative change
    for meteorological and hydrological variables.

    Parameters:
    var_names (list): Hydrological variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names
    MET_vars (list): Meteorological variable names e.g. ['precip', 'temp']
    mon_MED_drought (dict): {var: {dr: DataFrame}} monthly medians during drought
    mon_MED_relchange (dict): {var: {dr: DataFrame}} monthly relative change vs reference
    basin_avg (dict): {var: Series} long-term basin average per variable
    wmon_names (list): Water year month labels e.g. ['Oct', 'Nov', ...]
    dr_colors (dict): {dr: colour} per drought episode
    dr_markerstyles (dict): {dr: marker} per drought episode
    dr_linestyles (dict): {dr: linestyle} per drought episode
    var_plot_names (dict): {var: axis_label} for hydrological variables
    var_rel_names (dict): {var: short_name} for relative change axis labels
    met_plot_names (dict): {var: (axis_label, short_name)} for meteorological variables
    longterm_avg_start (int): Start year of long-term average for legend label (default: 1998)
    with_precip_temp (bool): Whether MET rows are included (default: True)
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save the figure to disk

    Returns:
    fig, axes
    """
    row_letters = ['a', 'b', 'c', 'd', 'e']
    fontsize    = 16
    n_rows      = len(MET_vars) + len(var_names) if with_precip_temp else len(var_names)
    figsize     = (20, 20) if with_precip_temp else (20, 12)

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=figsize)

    for i in range(n_rows):
        ax_abs = axes[i, 0]
        ax_rel = axes[i, 1]

        # Determine variable and labels for this row
        if with_precip_temp and i < len(MET_vars):
            var                 = MET_vars[i]
            plot_name, rel_name = met_plot_names[var]
        else:
            var_idx   = i - len(MET_vars) if with_precip_temp else i
            var       = var_names[var_idx]
            plot_name = var_plot_names[var]
            rel_name  = var_rel_names[var]

        # Left panel: absolute monthly medians
        for dr in dr_names:
            data = mon_MED_drought[var][dr]
            ax_abs.plot(
                wmon_names, data.median(axis=1),
                color=dr_colors[dr],
                label=f'{dr}, n:{len(data.columns)}',
                markersize=8, marker=dr_markerstyles[dr],
                linestyle=dr_linestyles[dr], linewidth=3
            )

        ax_abs.plot(
            wmon_names, basin_avg[var],
            label=f'{longterm_avg_start}-2022 Mean',
            color='black', linewidth=4
        )

        ax_abs.set_title(f'Monthly Median {plot_name}',  fontsize=fontsize + 2)
        ax_abs.set_ylabel(plot_name,                      fontsize=fontsize + 1)
        ax_abs.grid(axis='x', alpha=0.6)
        ax_abs.axhline(y=0.0, color='black', linestyle='-')
        ax_abs.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      prop={"size": fontsize - 2})

        # Right panel: relative change
        for dr in dr_names:
            data = mon_MED_relchange[var][dr]
            ax_rel.plot(
                wmon_names, data.median(axis=1),
                color=dr_colors[dr],
                label=f'{dr}, n:{len(data.columns)}',
                markersize=8, marker=dr_markerstyles[dr],
                linestyle=dr_linestyles[dr], linewidth=3
            )

        ax_rel.set_title(f'Monthly Median {rel_name} Relative Change (%)', fontsize=fontsize + 2)
        ax_rel.set_ylabel(f'{rel_name} Relative Change (%)',                fontsize=fontsize + 1)
        ax_rel.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax_rel.grid(axis='x', alpha=0.6)
        ax_rel.axhline(y=0.0, color='black', linestyle='-')

        for ax in [ax_abs, ax_rel]:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(fontsize)

        axes[i, 0].text(
            -0.15, 1.1, f'{row_letters[i]})',
            transform=axes[i, 0].transAxes,
            fontsize=fontsize + 4, fontweight='bold', va='top', ha='right'
        )

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = 'Combined_Seasonal_Analysis'
        fig.savefig(os.path.join(fig_dir, f'{save_name}.jpeg'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, f'{save_name}.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, axes


def plot_monthly_relchange_boxplots(var_names, dr_names, mon_MED_relchange, wmon_names,
                                    var_titles=None, bottom=-200, top=300,
                                    fig_dir=None, save=False):
    """
    Plot monthly relative change boxplots per variable, with one subplot per drought episode.

    Parameters:
    var_names (list): Variable names to plot e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names
    mon_MED_relchange (dict): {var: {dr: DataFrame}} monthly relative change vs reference
    wmon_names (list): Water year month labels e.g. ['Oct', 'Nov', ...]
    var_titles (dict): {var: display_name} for figure suptitle (default: {'RDC':'Q','WT':'WT','SC':'SC'})
    bottom (int): Lower y-axis limit for relative change (default: -200)
    top (int): Upper y-axis limit for relative change (default: 300)
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save figures to disk

    Returns:
    figs (dict): {var: fig} one figure per variable
    """
    if var_titles is None:
        var_titles = {var: var for var in var_names}

    figs = {}

    for var in var_names:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes      = axes.flatten()

        for ax, dr in zip(axes, dr_names):
            data         = mon_MED_relchange[var][dr]
            numbersites  = len(data.columns)

            data.transpose().plot(kind='box', ax=ax)

            # Flag months with values outside plot bounds
            out_of_bounds = ((data > top) | (data < bottom)).any(axis=1)
            custom_labels = [
                wmon_names[i] + ('*' if out_of_bounds.iloc[i] else '')
                for i in range(len(data.index))
            ]

            ax.set_title(f'Drought: {dr}, n: {numbersites}', fontsize=13)
            ax.set_xlabel('')
            ax.set_ylabel('Relative Change (%)', fontsize=12)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_ylim(bottom, top)
            ax.set_xticks(range(1, len(wmon_names) + 1))
            ax.set_xticklabels(custom_labels)
            ax.tick_params(axis='both', labelsize=12)

        fig.suptitle(
            f'{var_titles[var]} Median Monthly Relative Change',
            fontsize=15, y=0.99
        )
        fig.tight_layout()

        if save:
            if fig_dir is None:
                raise ValueError("fig_dir must be provided when save=True")
            fig.savefig(os.path.join(fig_dir, f'{var}_ALL_dr_ep_relchange_dist.jpeg'), dpi=300)
            fig.savefig(os.path.join(fig_dir, f'{var}_ALL_dr_ep_relchange_dist.svg'),
                        format='svg', transparent=True, dpi=300)
            print(f"Saved {var} figures to {fig_dir}")

        plt.show()
        figs[var] = fig

    return figs


def plot_mk_panel(ax, var, years, mk_trends, met_cmap, fontsize,
                  ylim=True, twin_ax=False):
    
    x = np.arange(1998, 2023, 1)
    all_sites_mean = years[var].mean(axis=1)
    ymax = all_sites_mean.max() + (all_sites_mean.max() * 0.3)
    handles = []
    if var == 'reservoirs':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plot_ax = ax
    
    if twin_ax:
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.tick_params(axis='y', colors=met_cmap[2])
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
        )
        for label in ax2.get_yticklabels():
            label.set_fontsize(fontsize + 1)
    else:
        ax2= None

    ## No trend
    nt = mk_trends[var]['notrend']
    if len(nt.index) > 0:
        nt_data      = years[var][nt.index]
        nt_trendline = np.arange(len(nt_data)) * nt['slope'].mean() + nt['intercept'].mean()
        nt_pval      = nt['p'].mean()

        l1, = plot_ax.plot(x, nt_data.mean(axis=1),
                           label=f'No sig. trend, n:{len(nt.index)}',
                           color=met_cmap[0], markersize=4, linewidth=2)
        l2, = plot_ax.plot(x, nt_trendline,
                           label=f'Trendline - slope:{round(nt["slope"].mean(), 3)}, p-val:{round(nt_pval, 3)}',
                           color=met_cmap[0], markersize=4, linewidth=2, linestyle=':')
        handles += [l1, l2]

    ## Increasing
    inc = mk_trends[var]['increasing']
    if len(inc.index) > 0:
        inc_data      = years[var][inc.index]
        inc_trendline = np.arange(len(inc_data)) * inc['slope'].mean() + inc['intercept'].mean()
        inc_pval      = inc['p'].mean()

        l3, = plot_ax.plot(x, inc_data.mean(axis=1),
                           label=f'Increasing sig. trend, n:{len(inc.index)}',
                           color=met_cmap[1], markersize=4, linewidth=2)
        l4, = plot_ax.plot(x, inc_trendline,
                           label=f'Trendline - slope:{round(inc["slope"].mean(), 3)}, p-val:{round(inc_pval, 3)}',
                           color=met_cmap[1], markersize=4, linewidth=2, linestyle=':')
        handles += [l3, l4]

    ## Decreasing
    dec = mk_trends[var]['decreasing']
    if len(dec.index) > 0:
        dec_data      = years[var][dec.index]
        dec_trendline = np.arange(len(dec_data)) * dec['slope'].mean() + dec['intercept'].mean()
        dec_pval      = dec['p'].mean()
        
        if ax2 != None:
            notrendmean = nt_data.mean().mean()
            decreasingmean = dec_data.mean().mean()
            if decreasingmean >= 1.5 * notrendmean:
                plot_ax = ax2 
            
        l5, = plot_ax.plot(x, dec_data.mean(axis=1),
                           label=f'Decreasing sig. trend, n:{len(dec.index)}',
                           color=met_cmap[2], markersize=4, linewidth=2)
        l6, = plot_ax.plot(x, dec_trendline,
                           label=f'Trendline - slope:{round(dec["slope"].mean(), 3)}, p-val:{round(dec_pval, 3)}',
                           color=met_cmap[2], markersize=4, linewidth=2, linestyle=':')
        handles += [l5, l6]

    ax.set_ylim(bottom =0)
    if ax2 != None: 
        ax.set_ylim(bottom =0)

    ybot, ytop = plot_ax.get_ylim()
    rect_height = ytop - ybot
    for start, width in [(2001, 1.5), (2012, 0.5), (2018, 0.5), (2020, 1.5)]:
        plot_ax.add_patch(mpl.patches.Rectangle(
            (start, ybot), width, rect_height,
            facecolor='pink', alpha=0.3, zorder=0
        ))

    ax.grid(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize + 1)

    return handles, ax2


def plot_combined_mk_trends(years, mk_trends, met_cmap, var_layout=None, legend_layout=None,
                             fontsize=10, fig_dir=None, save=False):
    """
    Plot combined Mann-Kendall trend panels for all variables with custom legend columns.

    Parameters:
    years (dict): {var: DataFrame} annual data per variable, used for site counts in legends
    mk_trends (dict): MK trend results passed through to plot_mk_panel
    met_cmap (list): Colourmap list passed through to plot_mk_panel
    var_layout (list): List of tuples defining each panel:
                       (row, col, var, title, ylabel, ylim, twin_ax)
                       If None, uses default 7-variable layout.
    legend_layout (list): List of tuples defining legend panels:
                          (row, vars_in_row, legend_titles)
                          If None, uses default layout derived from var_layout.
    fontsize (int): Base font size (default: 10)
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save the figure to disk

    Returns:
    fig, all_handles (dict): {var: list of line handles}
    """
    if var_layout is None:
        var_layout = [
            (0, 0, 'precip',            'a) Precip. Mann-Kendall Test',
             'Precip. (mm)',                          True,  False),
            (0, 1, 'temp',              'b) AT Mann-Kendall Test',
             r'Temp. ($^\circ$C)',                    False, False),
            (1, 0, 'runoff_efficiency', 'c) Runoff Efficiency Mann-Kendall Test',
             'Runoff Efficiency',                     False, False),
            (1, 1, 'RDC',              'd) Q Mann-Kendall Test',
             r'Q ($m^{3}.s^{-1}.km^{-2}$)',           False, False),
            (2, 0, 'WT',               'e) WT Mann-Kendall Test',
             r'WT ($^\circ$C)',                       True,  False),
            (2, 1, 'SC',               'f) SC Mann-Kendall Test',
             r'SC ($\mu$S.$cm^{-1}$)',                True,  False),
            (3, 0, 'reservoirs',       'g) Reservoir Storage Mann-Kendall Test',
             'Reservoir Storage (acre-feet)',          True,  True),
        ]

    if legend_layout is None:
        legend_layout = [
            (0, ['precip', 'temp'],
             [f"a) Precip. Legend, n:{len(years['precip'].columns)}",
              f"b) AT Legend, n:{len(years['temp'].columns)}"]),
            (1, ['runoff_efficiency', 'RDC'],
             [f"c) Runoff Efficiency Legend, n:{len(years['runoff_efficiency'].columns)}",
              f"d) Q Legend, n:{len(years['RDC'].columns)}"]),
            (2, ['WT', 'SC'],
             [f"e) WT Legend, n:{len(years['WT'].columns)}",
              f"f) SC Legend, n:{len(years['SC'].columns)}"]),
            (3, ['reservoirs'],
             [f"g) Reservoir Storage Legend, n:{len(years['reservoirs'].columns)}"]),
        ]

    fig         = plt.figure(figsize=(18, 22))
    gs          = fig.add_gridspec(4, 3, width_ratios=[2, 2, 1.2],
                                   hspace=0.15, wspace=0.25)
    all_handles = {}

    # Data panels
    for row, col, var, title, ylabel, ylim, twin_ax in var_layout:
        ax = fig.add_subplot(gs[row, col])

        handles, ax2 = plot_mk_panel(
            ax, var, years, mk_trends, met_cmap,
            fontsize=fontsize, ylim=ylim, twin_ax=twin_ax
        )
        all_handles[var] = handles

        ax.set_title(title,  fontsize=fontsize + 2)
        ax.set_ylabel(ylabel, fontsize=fontsize + 1)

        if twin_ax and ax2 is not None:
            ax2.set_ylabel(
                'Reservoir Storage (acre-feet)\nsites with decreasing trend',
                fontsize=fontsize, color=met_cmap[2]
            )

    # Legend panels (column 2)
    for row, vars_in_row, legend_titles in legend_layout:
        ax_leg = fig.add_subplot(gs[row, 2])
        ax_leg.axis('off')

        y_pos = 0.98
        for var, leg_title in zip(vars_in_row, legend_titles):
            ax_leg.text(
                0, y_pos, leg_title,
                fontsize=fontsize + 4, fontweight='bold',
                transform=ax_leg.transAxes, va='top'
            )
            y_pos -= 0.08

            for handle in all_handles[var]:
                label = handle.get_label()
                color = handle.get_color()
                ls    = handle.get_linestyle()
                ax_leg.plot([], [], color=color, linestyle=ls,
                            label=label, linewidth=2)
                ax_leg.text(
                    0.05, y_pos, label,
                    fontsize=fontsize + 2,
                    transform=ax_leg.transAxes, va='top', color=color
                )
                y_pos -= 0.07

            y_pos -= 0.05  # gap between variables

    fig.suptitle(
        'Long Term Annual Averages, water years 1998 - 2022',
        fontsize=fontsize + 4, y=0.91
    )

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'ALL_longterm_trend_MKtest_combined.jpeg'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'ALL_longterm_trend_MKtest_combined.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, all_handles


def get_facecolors(values, cmap, vmin, vmax, alpha=0.95):
    """
    Map scalar values → RGBA colors using cmap, clipped to [vmin, vmax].
    Returns an (N, 4) array of RGBA values with the given alpha.
    """
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba   = mapper.to_rgba(values, alpha=alpha)
    return rgba


def add_cbar_triangles(cbar, cmap, offset=0.07, tri_size=22):
    ax        = cbar.ax
    color_min = cmap(0.0)
    color_max = cmap(1.0)
    y_mid     = 0.5

    ax.plot(-offset, y_mid, marker='v', markersize=tri_size,
            color=color_min, clip_on=False,
            transform=ax.transAxes, zorder=5, linestyle='None')
    ax.plot(1 + offset, y_mid, marker='^', markersize=tri_size,
            color=color_max, clip_on=False,
            transform=ax.transAxes, zorder=5, linestyle='None')


def basin_boundary(upper_colorado_river_boundary_dir):
    fname         = upper_colorado_river_boundary_dir / 'Upper_Colorado_River_Basin_Boundary.shp'
    shape_feature = ShapelyFeature(shpreader(fname).geometries(),crs.PlateCarree(),edgecolor='dimgray',facecolor='white',lw=2 )
    return(shape_feature)


def _scatter_matched_edges(ax, d, marker, colorscheme, vmin, vmax,
                               linewidth=0.5, size=200, alpha=0.95):
        if len(d) == 0:
            return
        vals      = d['Relative Change (%)'].values
        facecolor = get_facecolors(vals, colorscheme, vmin, vmax, alpha=alpha)
        edgecolor        = facecolor.copy()
        edgecolor[:, :3] = np.clip(facecolor[:, :3] * 0.75, 0, 1)
        ax.scatter( x=d.LON, y=d.LAT, c=facecolor, s=size, alpha=None, marker=marker, edgecolors=edgecolor, linewidth=linewidth, transform=crs.PlateCarree(),  zorder=1 )

def _scatter_invisible(ax, d, colorscheme, vmin, vmax):
        return ax.scatter( x=d.LON, y=d.LAT, c=d['Relative Change (%)'], cmap=colorscheme, s=0, transform=crs.PlateCarree(), vmin=vmin, vmax=vmax )

    
## rel change maps with reservoirs
def plot_relchange_map(var_names, MED_relchange_map, var_settings,
                       upper_colorado_river_boundary_dir,
                       PUBS_sites_all3=None, wt_var=None,
                       extent=(-113, -105, 35.5, 43.5), reservoirs_mktrend = None,
                       fig_dir=None, save=False):

    if wt_var is None:
        wt_var = var_names[1]

    projection = crs.AlbersEqualArea(central_latitude=39.5, central_longitude=-98.35)

    fig = plt.figure(figsize=(26, 12))

    map_bottom  = 0.20
    map_height  = 0.73
    panel_width = 0.255
    gap         = 0.03
    left_margin = 0.04

    map_lefts = [left_margin,left_margin +     panel_width + gap,left_margin + 2 * (panel_width + gap),  ]

    map_axes = [
        fig.add_axes([l, map_bottom, panel_width, map_height],
                     projection=projection)
        for l in map_lefts
    ]

    # Colorbar axes
    cbar_height = 0.055
    cbar_bottom = 0.062
    cbar_width  = panel_width * 0.76

    cbar_q_left       = map_lefts[0] + (panel_width - cbar_width) / 2
    panels_1_2_centre = (map_lefts[1] + map_lefts[2] + panel_width) / 2
    cbar_shared_left  = panels_1_2_centre - cbar_width / 2

    cbar_ax_q      = fig.add_axes([cbar_q_left,      cbar_bottom, cbar_width, cbar_height])
    cbar_ax_shared = fig.add_axes([cbar_shared_left,  cbar_bottom, cbar_width, cbar_height])

    shape_feature = basin_boundary(upper_colorado_river_boundary_dir)

    shared_cbarplot = None

    for ax, var in zip(map_axes, var_names):
        s           = var_settings[var]
        colorscheme = s['colorscheme']
        vmin, vmax  = s['vmin'], s['vmax']
        data        = MED_relchange_map[var]
        alpha       = 0.95
        legend_handles = []
        ax.add_feature(cfeature.COASTLINE, linestyle=':', edgecolor='grey', linewidth=2)
        ax.add_feature(cfeature.STATES,    linestyle=':', edgecolor='grey', linewidth=2)
        ax.add_feature(cfeature.RIVERS)
        ax.set_extent(list(extent), crs=crs.PlateCarree())
        ax.add_feature(shape_feature, zorder=-1)

        is_wt_panel = (var == wt_var and PUBS_sites_all3 is not None)
        if is_wt_panel:
            valid_pubs = data.index.intersection(PUBS_sites_all3)
            plot_data  = data.drop(valid_pubs, errors='ignore')
            data_pubs  = data.loc[valid_pubs]
        else:
            plot_data = data
            data_pubs = None

        pos_data = plot_data[plot_data['Relative Change (%)'] >= 0]
        neg_data = plot_data[plot_data['Relative Change (%)'] <  0]
        _scatter_matched_edges(ax, pos_data, '^',colorscheme, vmin, vmax, linewidth=0.5, size=200)
        _scatter_matched_edges(ax, neg_data, 'v',colorscheme, vmin, vmax, linewidth=0.5, size=200)

        if is_wt_panel and data_pubs is not None and len(data_pubs) > 0:
            pos_pubs = data_pubs[data_pubs['Relative Change (%)'] >= 0]
            neg_pubs = data_pubs[data_pubs['Relative Change (%)'] <  0]
            _scatter_matched_edges(ax, pos_pubs, '*',colorscheme, vmin, vmax, linewidth=0.5, size=400)
            _scatter_matched_edges(ax, neg_pubs, '*',colorscheme, vmin, vmax, linewidth=0.5, size=400)
         
        ax.set_title(
            f"{s['title_label']} {s['plot_name']}, n:{len(data.index)}",fontsize=22 )
        
        dref = data[data['CLASS'] == 'Ref']
        if len(dref) > 0:
            ax.scatter(x=dref.LON, y=dref.LAT, c='black',s=100, alpha=0.95, marker='.', linewidth=0.5, transform=crs.PlateCarree(), zorder=2)
        
        cbarplot = _scatter_invisible(ax, data,colorscheme, vmin, vmax)
        if var == wt_var:
            shared_cbarplot = cbarplot
            
        if reservoirs_mktrend is not None:
            res_notrend_handle = None
            res_decreasing_handle = None
            res_increasing_handle = None
            for MK_type in ['notrend','increasing','decreasing']:
                data = reservoirs_mktrend[MK_type]
                if MK_type == 'notrend': 
                    res_notrend_handle = ax.scatter(x=data.LON, y=data.LAT, c='black',s= 50, alpha=1, label = 'Reservoirs: No Trend',
                            marker= 's', #edgecolors= (0, 0, 1, 0.1), #(0.5, 0, 0.5, 0.1), # edgecolors=(0.5, 0.5, 0.5, 1) edgecolors=(0, 0, 1, 0.1),
                            linewidth=0.5,transform=crs.PlateCarree(),vmin = vmin, vmax = vmax,zorder=1)
                if MK_type == 'decreasing': 
                    res_decreasing_handle = ax.scatter(x=data.LON, y=data.LAT, c='forestgreen',s= 350, alpha=0.8,label = 'Reservoirs: Decreasing',
                            marker= 'v', #edgecolors= (1, 0, 0, 0.1), #(0.5, 0, 0.5, 0.1), # edgecolors=(0.5, 0.5, 0.5, 1) edgecolors=(0, 0, 1, 0.1),
                            linewidth=0.5,transform=crs.PlateCarree(),vmin = vmin, vmax = vmax,zorder=1)   
                if MK_type == 'increasing': 
                    res_increasing_handle = ax.scatter(x=data.LON, y=data.LAT, c='gold',s= 350, alpha=0.99,label = 'Reservoirs: Increasing',
                            marker= '^', #edgecolors= (0, 0, 1, 0.1), #(0.5, 0, 0.5, 0.1), # edgecolors=(0.5, 0.5, 0.5, 1) edgecolors=(0, 0, 1, 0.1),
                            linewidth=0.5,transform=crs.PlateCarree(),vmin = vmin, vmax = vmax,zorder=1)
                #legend_handles += [res_notrend_handle, res_decreasing_handle,res_increasing_handle]
                #ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
            legend_handles += [h for h in [res_notrend_handle, res_decreasing_handle, res_increasing_handle] if h is not None]

        
        if var in [var_names[0], var_names[1]]:
            dot_handle = mlines.Line2D([], [], color='black', marker='.', linestyle='None',markersize=10, label='Pristine site')
            legend_handles += [dot_handle]

            if var == wt_var and PUBS_sites_all3 is not None:
                star_handle = mlines.Line2D([], [], color='grey', marker='*', linestyle='None',markeredgecolor='grey', markeredgewidth=0.5,
                    markersize=13, label='PUBS site' )
                tri_handle = mlines.Line2D([], [], color='grey', marker='^', linestyle='None',markeredgecolor='grey', markeredgewidth=0.5,
                    markersize=10, label='Non-PUBS site' )
                legend_handles += [star_handle, tri_handle]
        if legend_handles != []:
            ax.legend(handles=legend_handles,loc='upper left',fontsize=13,framealpha=0.85,labelspacing=0.6,handletextpad=0.5,borderpad=0.6,)

        if var == var_names[0]:
            cbar_q = fig.colorbar(cbarplot, cax=cbar_ax_q,extend='neither',orientation='horizontal' )
            cbar_q.ax.tick_params(labelsize=15)          # back to previous size
            cbar_q.set_label(s['cbar_title'], fontsize=16, labelpad=10)
            add_cbar_triangles(cbar_q, colorscheme, offset=0.07, tri_size=22)
            
    if shared_cbarplot is not None:
        s_wt = var_settings[wt_var]
        cbar_shared = fig.colorbar(shared_cbarplot, cax=cbar_ax_shared,extend='neither',orientation='horizontal')
        if s_wt['cbar_ticks'] is not None:
            cbar_shared.set_ticks(s_wt['cbar_ticks'])
            cbar_shared.set_ticklabels([str(t) for t in s_wt['cbar_ticks']])
        cbar_shared.ax.tick_params(labelsize=15)         # back to previous size
        cbar_shared.set_label(s_wt['cbar_title'], fontsize=16, labelpad=10)
        add_cbar_triangles(cbar_shared, colorscheme, offset=0.07, tri_size=22)
    
    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        if reservoirs_mktrend is None:
            save_name = 'ALL3_locations_Wpristine'
        else:
            save_name = 'ALL3_locations_Wpristine_Wreservoirs'
        fig.savefig(os.path.join(fig_dir, save_name+'.jpeg'),
                    format='jpeg', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, save_name+'.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, map_axes

def plot_colocation_relchange(var_names, dr_names, all3_ann_MEDrelchange_allsites, all_sites,all3vars, MED_relchange_map, upper_colorado_river_boundary_dir,
                              var_markers= None,var_labels= None,var_colors=None, ylim = (-60, 90),extent= (-113, -105, 35.5, 43.5),fontsize = 16,fig_dir = None,
                              save = False ):
    ## Plot
    #var_markers = {var_names[0]: 'v', var_names[1]: 'o', var_names[2]: 'x'}
    if var_markers == None:
        var_markers = {var: c for var, c in zip(var_names, ['^', '*', 'o'])}
    if var_colors == None:
        var_colors = {var: c for var, c in zip(var_names, ['steelblue', 'tomato', 'goldenrod'])}
    if var_labels == None:
        var_labels  = {var_names[0]: 'Q', var_names[1]: 'WT', var_names[2]: 'SC'}

    fig = plt.figure(figsize=(12, 16))
    gs  = fig.add_gridspec(4, 2, width_ratios=[4, 1], hspace=0.4, wspace=0.1)
    shape_feature = basin_boundary(upper_colorado_river_boundary_dir)

    x_coords  = np.arange(len(all_sites))
    
    for i, dr in enumerate(dr_names):
        # Left panel - scatter line plot
        ax_line = fig.add_subplot(gs[i, 0])
        ax_line.set_title(f'Drought: {dr}', fontsize=fontsize + 2)

        for var in var_names:
            series = all3_ann_MEDrelchange_allsites[dr][var].reindex(all_sites)
            ax_line.scatter(x_coords, series.values, s=70, marker=var_markers[var],
                        label=var_labels[var], color=var_colors[var])

        ax_line.set_ylabel('Relative Change (%)', fontsize=fontsize)
        ax_line.set_xlim(-0.5, len(all_sites) - 0.5)
        ax_line.set_xticks(x_coords)
        ax_line.set_xticklabels([str(s) for s in all_sites], rotation=15, fontsize=fontsize - 4)
        ax_line.set_ylim(-60, 90)
        ax_line.axhline(y=0.0, color='black', linestyle='-')
        ax_line.grid(True, alpha=0.6)
        for label in ax_line.get_yticklabels():
            label.set_fontsize(fontsize - 4)

        ax_line.text(-0.12, 1.05, f'{chr(97+i)})', transform=ax_line.transAxes,
                 fontsize=fontsize + 4, fontweight='bold', va='top', ha='right')

        # Right panel - map
        ax_map = fig.add_subplot(gs[i, 1], projection=crs.PlateCarree())
        ax_map.add_feature(cfeature.COASTLINE, linestyle=':', edgecolor='grey', linewidth=1)
        ax_map.add_feature(cfeature.STATES,    linestyle=':', edgecolor='grey', linewidth=1)
        ax_map.add_feature(cfeature.RIVERS,    edgecolor='blue', linewidth=0.5)
        ax_map.add_feature(shape_feature, zorder=-1)
        ax_map.set_extent(extent, crs=crs.PlateCarree())

        data = MED_relchange_map[var_names[0]].loc[all3vars[dr]]
        ax_map.scatter(x=data.LON, y=data.LAT, c='black', s=75, alpha=0.8,
                   marker='X', linewidth=0.5, transform=crs.PlateCarree(), zorder=2)

        dref = data[data['CLASS'] == 'Ref']
        if len(dref) > 0:
            ax_map.scatter(x=dref.LON, y=dref.LAT, c='black', s=100, alpha=0.8,
                       marker='.', linewidth=0.5, transform=crs.PlateCarree(), zorder=2)

        for lon, lat, name in zip(data['LON'], data['LAT'], data.index):
            offset, ha = (6, 6), 'center'
            if name == 9144250:   offset, ha = (6, -3),   'left'
            elif name == 9149500: offset, ha = (6, -11),  'left'
            elif name == 9152500: offset, ha = (6, 0),    'left'
            elif name in [9171100, 9169500]: offset, ha = (0, -12), 'center'
            elif name == 9180000: offset, ha = (-6, -3),  'right'
            ax_map.annotate(name, (lon, lat), textcoords="offset points",
                        xytext=offset, ha=ha, fontsize=9, color='black')

        if i == 0:
            ax_line.legend(loc='upper right', bbox_to_anchor=(1.0, 0.99),
                       ncol=3, fontsize=fontsize, frameon=True)

    plt.tight_layout()
    
    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'combined_annual_relchange_CoLoc.jpeg'),
                    format='jpeg', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'combined_annual_relchange_CoLoc.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig

    
def plot_colocation_seasonal_relchange(var_names,MET_vars, dr_names, all3_ann_MEDrelchange, all_sites, mon_MED_relchange, wmon_names,
                                       dr_markerstyles,dr_colors,dr_linestyles, ylim = (-60, 90),extent= (-113, -105, 35.5, 43.5),fontsize = 16,
                                       with_precip_temp=True, same_yaxis = False, fig_dir = None, save = False ):

    title_label = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ']

    figsize = (10, 20) if with_precip_temp else (10, 10)
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=figsize)

    var_plot_names = {var_names[0]: 'Q', var_names[1]: 'WT', var_names[2]: 'SC'}
    met_plot_names = {MET_vars[0]: 'P', MET_vars[1]: 'AT'}

    for i, ax in enumerate(axs):

        if i in (0, 1):
            # --- Meteorological panels ---
            met2plot = MET_vars[i]
            met_plot_name = met_plot_names[met2plot]

            ax.set_ylabel(f'{met_plot_name} Relative Change (%)', fontsize=fontsize + 1)
            ax.set_title(f'{title_label[i]}Monthly Median {met_plot_name} Relative Change (%)', fontsize=fontsize + 2)

            handles = []
            for dr in dr_names:
                colocated_sites = all3_ann_MEDrelchange[dr].index.astype(mon_MED_relchange['RDC'][dr].columns.dtype)
                data = mon_MED_relchange[met2plot][dr][colocated_sites]
                line, = ax.plot(wmon_names,data.median(axis=1),color=dr_colors[dr],label=f'{dr}, n:{len(data.columns)}',markersize=8,
                                marker=dr_markerstyles[dr],linestyle=dr_linestyles[dr], linewidth=3)
                handles.append(line)

        elif i in (2, 3, 4):
            # --- Hydro variable panels ---
            var2_plot = var_names[i - 2]
            var_plot_name = var_plot_names[var2_plot]

            ax.set_ylabel(f'{var_plot_name} Relative Change (%)', fontsize=fontsize + 1)
            ax.set_title(f'{title_label[i]}Monthly Median {var_plot_name} Relative Change', fontsize=fontsize + 2)

            handles = []
            for dr in dr_names:
                colocated_sites = all3_ann_MEDrelchange[dr].index.astype(mon_MED_relchange[var2_plot][dr].columns.dtype)
                data = mon_MED_relchange[var2_plot][dr][colocated_sites]
                line, = ax.plot(wmon_names, data.median(axis=1), color=dr_colors[dr], label=f'{dr}, n:{len(data.columns)}', markersize=8,
                                marker=dr_markerstyles[dr], linestyle=dr_linestyles[dr], linewidth=3  )
                handles.append(line)

        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.axhline(y=0.0, color='black', linestyle='-')
        ax.grid(axis='x', alpha=0.6)
        if i == 4:
            ax.legend(bbox_to_anchor=(0.1, 1.0), handles=handles, loc='upper left', prop={"size": fontsize})
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontsize + 1)
        if same_yaxis:
            ax.set_ylim(bottom=-150, top=175)

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        if with_precip_temp:
            save_name = 'ALL3seasonal_relchangeWY_colocated'
        else: 
            save_name = 'ALL3seasonal_relchangeWY_noPT_colocated'
        fig.savefig(os.path.join(fig_dir, f'{save_name}.jpeg'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, f'{save_name}.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, axs


def plot_relchange_percentile_maps(var_names, MED_relchange_map, pct_map,
                                   upper_colorado_river_boundary_dir,colorschemes,
                                   title_labels=None, vmin_vmax=None,
                                   PUBS_sites_all3=None, wt_var=None,
                                   extent=(-113, -105, 35.5, 43.5),
                                   fig_dir=None, save=False):

    if wt_var is None:
        wt_var = var_names[1]

    if title_labels is None:
        letters      = 'abcdefghijklmnopqrstuvwxyz'
        title_labels = {}
        for v_idx, var in enumerate(var_names):
            base = v_idx * 3
            title_labels[var] = {
                'Median': f'{letters[base]})',
                '95':     f'{letters[base + 1]})',
                '5':      f'{letters[base + 2]})',
            }

    if vmin_vmax is None:
        vmin_vmax = {var: (-100, 100) for var in var_names}

    projection = crs.AlbersEqualArea(central_latitude=39.5, central_longitude=-98.35)

    n_vars  = len(var_names)   # rows
    n_cols  = 3                # columns: Median, 95th, 5th

    fig = plt.figure(figsize=(26, 9 * n_vars))

    left_margin = 0.01
    panel_width = 0.25
    gap         = 0.005
    map_height  = 0.8 / n_vars
    row_gap     = 0.05          # vertical gap between rows (holds colorbars)

    col_lefts = [
        left_margin,
        left_margin +     panel_width + gap,
        left_margin + 2 * (panel_width + gap),
    ]

    # Row bottoms: top row = index 0, bottom row = index n_vars-1
    # Leave space at the very bottom for the two colorbars
    cbar_zone   = 0.10   
    row_bottoms = [
        cbar_zone + (n_vars - 1 - i) * (map_height + row_gap)
        for i in range(n_vars)
    ]

    # map_axes[i][j] = row i, col j
    map_axes = [
        [fig.add_axes([col_lefts[j], row_bottoms[i], panel_width, map_height],
                      projection=projection)
         for j in range(n_cols)]
        for i in range(n_vars)
    ]

    shape_feature = basin_boundary(upper_colorado_river_boundary_dir)

    # Two colorbars: Q (row 0) and shared WT+SC (rows 1+)

    cbar_height = 0.035
    cbar_bottom = cbar_zone * 0.25          # a bit above the figure bottom
    cbar_width  = panel_width * 0.76

    cbar_q_left      = col_lefts[0] + (panel_width - cbar_width) / 2
    panels_1_2_centre = (col_lefts[1] + col_lefts[2] + panel_width) / 2
    cbar_shared_left  = panels_1_2_centre - cbar_width / 2

    cbar_ax_q      = fig.add_axes([cbar_q_left,     cbar_bottom, cbar_width, cbar_height])
    cbar_ax_shared = fig.add_axes([cbar_shared_left, cbar_bottom, cbar_width, cbar_height])

    cbar_anchor_q      = None
    cbar_anchor_shared = None
    q_var              = var_names[0]

    for i, var in enumerate(var_names):
        colorscheme = colorschemes[var]
        vmin, vmax  = vmin_vmax[var]
        labels      = title_labels[var]

        col_data   = [MED_relchange_map[var], pct_map[var]['95'], pct_map[var]['5']]
        col_titles = [
            f"{labels['Median']} {var} Median",
            f"{labels['95']} {var} 95th Percentile",
            f"{labels['5']} {var} 5th Percentile",
        ]

        for j, (ax, data, title) in enumerate(zip(map_axes[i], col_data, col_titles)):

            ax.add_feature(cfeature.COASTLINE, linestyle=':', edgecolor='grey', linewidth=2)
            ax.add_feature(cfeature.STATES,    linestyle=':', edgecolor='grey', linewidth=2)
            ax.add_feature(cfeature.RIVERS)
            ax.set_extent(list(extent), crs=crs.PlateCarree())
            ax.add_feature(shape_feature, zorder=-1)

            is_wt_panel = False
            if var == wt_var:
                is_wt_panel = True 
            if is_wt_panel:
                valid_pubs = data.index.intersection(PUBS_sites_all3)
                plot_data  = data.drop(valid_pubs, errors='ignore')
                data_pubs  = data.loc[valid_pubs]
            else:
                plot_data = data
                data_pubs = None

            pos_data = plot_data[plot_data['Relative Change (%)'] >= 0]
            neg_data = plot_data[plot_data['Relative Change (%)'] <  0]
            _scatter_matched_edges(ax, pos_data, '^', colorscheme, vmin, vmax,
                                   linewidth=0.5, size=200)
            _scatter_matched_edges(ax, neg_data, 'v', colorscheme, vmin, vmax,
                                   linewidth=0.5, size=200)

            if is_wt_panel and data_pubs is not None and len(data_pubs) > 0:
                pos_pubs = data_pubs[data_pubs['Relative Change (%)'] >= 0]
                neg_pubs = data_pubs[data_pubs['Relative Change (%)'] <  0]
                _scatter_matched_edges(ax, pos_pubs, '*', colorscheme, vmin, vmax,
                                       linewidth=0.5, size=400)
                _scatter_matched_edges(ax, neg_pubs, '*', colorscheme, vmin, vmax,
                                       linewidth=0.5, size=400)

            dref = data[data['CLASS'] == 'Ref']
            if len(dref) > 0:
                ax.scatter(x=dref.LON, y=dref.LAT, c='black',s=100, alpha=0.95, marker='.', linewidth=0.5,transform=crs.PlateCarree(), zorder=2)

            if j == 0:
                if var != var_names[2]:
                    dot_handle = mlines.Line2D([], [], color='black', marker='.', linestyle='None',markersize=10, label='Pristine site')
                    legend_handles = [dot_handle]

                    if is_wt_panel:
                        star_handle = mlines.Line2D([], [], color='grey', marker='*', linestyle='None', markeredgecolor='grey', markeredgewidth=0.5,
                            markersize=13, label='PUBS site')
                        tri_handle  = mlines.Line2D([], [], color='grey', marker='^', linestyle='None', markeredgecolor='grey', markeredgewidth=0.5,
                            markersize=10, label='Non-PUBS site')
                        legend_handles += [star_handle, tri_handle]

                    ax.legend(handles=legend_handles,loc='upper left',fontsize=13,framealpha=0.85,labelspacing=0.6,handletextpad=0.5,borderpad=0.6,)

            ax.set_title(f'{title}, n:{len(data.index)}', fontsize=22)

            cbarplot = _scatter_invisible(ax, data, colorscheme, vmin,vmax)
            if var == q_var and cbar_anchor_q is None:
                cbar_anchor_q = cbarplot
            if var == wt_var and cbar_anchor_shared is None:
                cbar_anchor_shared = cbarplot

    if cbar_anchor_q is not None:
        cbar_q = fig.colorbar(cbar_anchor_q, cax=cbar_ax_q,extend='neither', orientation='horizontal')
        cbar_q.ax.tick_params(labelsize=13)
        cbar_q.set_label('Q Relative Change (%)', fontsize=16, labelpad=10)
        add_cbar_triangles(cbar_q, colorschemes[q_var], offset=0.07, tri_size=22)

    if cbar_anchor_shared is not None:
        cbar_shared = fig.colorbar(cbar_anchor_shared, cax=cbar_ax_shared,extend='neither', orientation='horizontal' )
        cbar_shared.ax.tick_params(labelsize=13)
        cbar_shared.set_label('WT & SC Relative Change (%)', fontsize=16, labelpad=10)
        add_cbar_triangles(cbar_shared, colorschemes[wt_var], offset=0.07, tri_size=22)

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'ALL3_locations_percentiles_Wpristine.jpeg'),
                    format='jpeg', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'ALL3_locations_percentiles_Wpristine.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, map_axes


def plot_pq_scatter_ex_sites(sites_plot, runoff_boxcox, P_years, I_all, results,
                                   panel_labels=None, fig_dir=None, save=False):
    """
    Scatterplot of Box-Cox transformed runoff vs precipitation for example sites,
    with regression lines per drought/non-drought period using adj_a0 pre-whitening correction.

    Parameters:
    sites_plot (list): Site IDs to plot, one per panel (length determines nrows)
    runoff_boxcox (dict): {site: Series} Box-Cox transformed runoff per site
    P_years (dict): {site: Series} annual precipitation per site
    I_all (dict): {site: Series} drought indicator (0=non-drought, 1=drought) per site
    results (DataFrame): Regression results with index of site IDs and columns:
                         'a0', 'a1', 'a2', 'rho', 'pval_a1'
    panel_labels (list): Panel letter labels e.g. ['a)', 'b)', 'c)'].
                         If None, auto-generates from alphabet.
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save the figure to disk

    Returns:
    fig, axes
    """
    if panel_labels is None:
        letters      = 'abcdefghijklmnopqrstuvwxyz'
        panel_labels = [f'{letters[i]})' for i in range(len(sites_plot))]

    fsize     = 12
    fontsize2 = fsize - 2
    n_panels  = len(sites_plot)

    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(6, n_panels * 10 / 3))

    # Ensure axes is always iterable even for a single panel
    if n_panels == 1:
        axes = [axes]

    for val, ax in enumerate(axes):
        site = sites_plot[val]

        df = pd.DataFrame({
            'Runoff': runoff_boxcox[site],
            'P':      P_years[site],
            'I':      I_all[site]
        }).dropna()

        # Fetch regression results — handle int or string index
        row = results.loc[int(site)] if results.index.dtype != object else results.loc[site]
        a0, a1, a2, rho = row['a0'], row['a1'], row['a2'], row['rho']
        pval            = row['pval_a1']

        # Project pre-whitened intercept back into Box-Cox data space
        adj_a0 = a0 / (1 - rho) if (not np.isnan(rho) and abs(rho) < 1) else a0

        period_styles = {0: ('Non-drought', 'blue'), 1: ('Drought', 'red')}

        for i_val, (label, color) in period_styles.items():
            subset = df[df['I'] == i_val]
            if subset.empty:
                continue

            ax.scatter(
                x=subset['P'], y=subset['Runoff'],
                c=color, label=label, alpha=0.7, s=20
            )

            P_sorted = subset['P'].sort_values()
            Q_line   = adj_a0 + a1 * i_val + a2 * P_sorted
            ax.plot(P_sorted, Q_line, color=color)

        if pval >= 0.05:
            title = f'{panel_labels[val]} Not Significant (p={pval:.3f}), Site {site}'
        else:
            direction = 'Pos.' if a1 >= 0 else 'Neg.'
            title     = f'{panel_labels[val]} {direction} Significant (p={pval:.3f}), Site {site}'

        ax.set_title(title,            fontsize=fsize)
        ax.set_xlabel('P (mm)',        fontsize=fsize)
        ax.set_ylabel('Runoff (Box-Cox)', fontsize=fsize)
        ax.legend(loc='best')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fontsize2)

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        fig.savefig(os.path.join(fig_dir, 'p_q_examplesites_adjintercept.jpeg'), dpi=300)
        fig.savefig(os.path.join(fig_dir, 'p_q_examplesites_adjintercept.svg'),
                    format='svg', transparent=True, dpi=300)
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, axes
    
    
def partition_reservoirs(reservoirs_data, start='1997-10', end='2022-09-30',
                         top3_cols=None, mid_threshold=100000,
                         manual_mid_add=None, manual_mid_remove=None):
    """
    Partition reservoir data into top3, mid-range, and rest groups.

    Returns
    -------
    dict with keys: 'all', 'top3', 'nopowell', 'notop3', 'mid', 'rest'
    and sort orders preserved across groups for color consistency.
    """
    if top3_cols is None:
        top3_cols = ['AZlakepowell', 'UTflaminggorge', 'NMnavajo']
    if manual_mid_add is None:
        manual_mid_add = ['COcrawford', 'COhomestake']
    if manual_mid_remove is None:
        manual_mid_remove = ['COcrawford', 'COhomestake']

    df = reservoirs_data.set_index('datetime').sort_index()
    df = df.loc[start:end]

    top3    = df[top3_cols]
    nopowell = df.drop(columns=[top3_cols[0]])
    notop3   = nopowell.drop(columns=top3_cols[1:])

    site_means = notop3.mean().sort_values(ascending=False)
    mid_cols   = list(site_means[site_means > mid_threshold].index)
    for col in manual_mid_add:
        if col not in mid_cols:
            mid_cols.append(col)
    mid  = notop3[mid_cols]

    rest_cols = list(site_means[site_means <= mid_threshold].index)
    for col in manual_mid_remove:
        if col in rest_cols:
            rest_cols.remove(col)
    rest = notop3[rest_cols]

    # Preserve sort order for color consistency
    reservoirlist_notop3  = list(notop3.columns)
    reservoirlist_nopowell = reservoirlist_notop3 + top3_cols[1:]
    reservoirlist_all      = reservoirlist_nopowell + [top3_cols[0]]

    def sorted_subset(df, order):
        cols = [c for c in order if c in df.columns]
        return df[cols]

    return {
        'all':     sorted_subset(df,       reservoirlist_all),
        'top3':    top3,
        'nopowell':sorted_subset(nopowell, reservoirlist_nopowell),
        'notop3':  sorted_subset(notop3,   reservoirlist_notop3),
        'mid':     mid,
        'rest':    rest,
    }


def prep_reservoir_line_map(reservoirs_metadata,reservoirs_data,mk_trends_reservoirs):
    if 'STORAGE_DATA_NAME' in reservoirs_metadata.columns:
        reservoirs_metadata.set_index('STORAGE_DATA_NAME',inplace=True)
    res_partitions = partition_reservoirs(reservoirs_data)
    res_names = {}

    for partition in ['top3', 'mid',  'rest']:
        name_dict = reservoirs_metadata.loc[res_partitions[partition].columns]['site_metadata.site_name'].to_dict()
        res_names[partition] = name_dict
    
    reservoirs_mktrend = {}
    for trend in ['notrend','increasing','decreasing']:
        # mk_trends_reservoirs = mk_trends['reservoirs']
        reservoirs_mktrend[trend] = mk_trends_reservoirs[trend][['trend']]

        reservoirs_mktrend[trend]['LAT'] = 0.0
        reservoirs_mktrend[trend]['LON'] = 0.0
        reservoirs_mktrend[trend]['elevation'] = 0.0
        reservoirs_mktrend[trend]['name'] = ''
        for site in reservoirs_mktrend[trend].index:
            reservoirs_mktrend[trend].at[site, 'LAT'] = reservoirs_metadata.loc[site,'site_metadata.lat']
            reservoirs_mktrend[trend].at[site, 'LON'] = reservoirs_metadata.loc[site,'site_metadata.longi'] 
            reservoirs_mktrend[trend].at[site, 'elevation'] = reservoirs_metadata.loc[site,'site_metadata.elevation'] 
            reservoirs_mktrend[trend].at[site, 'name'] = reservoirs_metadata.loc[site,'site_metadata.site_name']
        
        reservoirs_mktrend[trend]['LAT'] = reservoirs_mktrend[trend]['LAT'].astype(float)
        reservoirs_mktrend[trend]['LON'] = reservoirs_mktrend[trend]['LON'].astype(float)
        reservoirs_mktrend[trend]['elevation'] = reservoirs_mktrend[trend]['elevation'].astype(float)


    # --- All label offsets in one dict (defaults to (6,6) center if not listed) ---
    label_offsets = {
    'STARVATION RESERVOIR':        dict(xytext=(-6, -12), ha='left'),
    'BLUE MESA RESERVOIR':         dict(xytext=(-6, -12), ha='left'),
    'EDEN RESERVOIR':              dict(xytext=(-6, -12), ha='left'),
    'STEINAKER RESERVOIR':         dict(xytext=(-6, -12), ha='left'),
    'LAKE NIGHTHORSE':             dict(xytext=(-6, -12), ha='left'),
    'VALLECITO RESERVOIR':         dict(xytext=(-6, -12), ha='left'),
    'HOMESTAKE RESERVOIR':         dict(xytext=(-5, -3),  ha='right'),
    'MEEKS CABIN RESERVOIR':       dict(xytext=(-5, -3),  ha='right'),
    'JOES VALLEY RESERVOIR':       dict(xytext=(-5, -3),  ha='right'),
    'CURRANT CREEK RESERVOIR':     dict(xytext=(-5, -3),  ha='right'),
    'JACKSON GULCH RESERVOIR':     dict(xytext=(-5, -3),  ha='right'),
    'UPPER STILLWATER RESERVOIR':  dict(xytext=(-5, -3),  ha='right'),
    'VEGA RESERVOIR':              dict(xytext=(-5, -3),  ha='right'),
    'FRUITGROWERS RESERVOIR':      dict(xytext=(-5, -3),  ha='right'),
    'CRYSTAL RESERVOIR':           dict(xytext=(-5, -3),  ha='right'),
    'RIDGWAY RESERVOIR':           dict(xytext=(-5, -3),  ha='right'),
    'RUEDI RESERVOIR':             dict(xytext=(5, -2.5), ha='left'),
    'PAONIA RESERVOIR':            dict(xytext=(5, -2.5), ha='left'),
    'WILLIAMS FORK RESERVOIR':     dict(xytext=(5, -2.5), ha='left'),
    'RIFLE GAP RESERVOIR':         dict(xytext=(5, -2.5), ha='left'),
    'STATELINE RESERVOIR':         dict(xytext=(5, -2.5), ha='left'),
    'RED FLEET RESERVOIR':         dict(xytext=(5, -2.5), ha='left'),
    'SILVER JACK RESERVOIR':       dict(xytext=(5, -2.5), ha='left'),
    'GREEN MOUNTAIN RESERVOIR':    dict(xytext=(5, -5),   ha='left'),
    'TAYLOR PARK RESERVOIR':       dict(xytext=(5, -5),   ha='left'),
    'SHADOW MOUNTAIN RESERVOIR':   dict(xytext=(1, 9),    ha='left'),
    'MORROW POINT RESERVOIR':      dict(xytext=(-5, -5),  ha='right'),
    'GRAND LAKE':                  dict(xytext=(5, 2.5),  ha='left'),
    'WILLOW CREEK RESERVOIR':      dict(xytext=(5, -1),   ha='left'),
    }

    drought_periods = [('2000-10-01', 730),('2012-01-01', 365), ('2017-10-01', 365), ('2019-10-01', 730), ]

    mk_styles = {
    'notrend':    dict(c='black', marker='.',  s=75, alpha=0.8, linewidth=0.5),
    'increasing': dict(c='blue',  marker='^',  s=75, alpha=0.8, linewidth=0.5),
    'decreasing': dict(c='red',   marker='v',  s=75, alpha=0.8, linewidth=0.5),
    }

    row_config = [('top3', 'Top 3 Reservoirs'),('mid',  'Mid-Range Reservoirs'), ('rest', 'Smaller Reservoirs'),]

    return res_partitions, reservoirs_mktrend, label_offsets, drought_periods, mk_styles,row_config,res_names


## Reservoir storage
def plot_reservoir_line_map(reservoirs_metadata, reservoirs_data, mk_trends_reservoirs,
                           upper_colorado_river_boundary_dir, fig_dir = None, save = False):
    res_partitions, reservoirs_mktrend, label_offsets, drought_periods, mk_styles, row_config, res_names = prep_reservoir_line_map(reservoirs_metadata,reservoirs_data,mk_trends_reservoirs)

    fontsize = 14
    fig = plt.figure(figsize=(18, 22))
    gs  = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], hspace=0.35, wspace=0.1)

    shape_feature = basin_boundary(upper_colorado_river_boundary_dir)
    
    detailed_rivers = cfeature.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines',scale='10m', facecolor='none', edgecolor='blue' )

    fig.suptitle('UCRB Reservoir Storage (acre-feet), 1998 – 2022', fontsize=fontsize + 2, y=0.9)

    for row_idx, (group_key, title) in enumerate(row_config):
        ax_ts  = fig.add_subplot(gs[row_idx, 0])
        ax_map = fig.add_subplot(gs[row_idx, 1], projection=crs.PlateCarree())

        res_df = res_partitions[group_key].copy()
        res_df.index = pd.to_datetime(res_df.index)

        # Time series 
        handles = []
        for col in res_df.columns:
            line, = ax_ts.plot(res_df.index, res_df[col], label=res_names[group_key][col], linewidth=1.5)
            handles.append(line)

        ymax = res_df.max().max() * 1.05
        ax_ts.set_ylim(0, ymax)
        ax_ts.set_title(title, fontsize=fontsize)
        ax_ts.set_ylabel('Reservoir Storage\n(acre-feet)', fontsize=fontsize)
        ax_ts.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        ax_ts.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
        ax_ts.xaxis.set_major_locator(mpl.dates.YearLocator(2))
        ax_ts.tick_params(axis='x', rotation=0)
        ax_ts.grid(False)
        for label in ax_ts.get_xticklabels() + ax_ts.get_yticklabels():
            label.set_fontsize(fontsize - 2)

        for start_str, days in drought_periods:
            start_dt = pd.Timestamp(start_str)
            ax_ts.axvspan(start_dt, start_dt + pd.Timedelta(days=days),
                      ymin=0, ymax=1, facecolor='pink', alpha=0.3, zorder=0)

        ax_ts.legend(handles=handles, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncols=4,
                 prop={'size': fontsize - 4}, frameon=False)

        #  Map 
        ax_map.add_feature(cfeature.COASTLINE, linestyle=':', edgecolor='grey', linewidth=1)
        ax_map.add_feature(cfeature.STATES,    linestyle=':', edgecolor='grey', linewidth=1)
        ax_map.add_feature(detailed_rivers,    edgecolor='blue', linewidth=0.5)
        ax_map.add_feature(shape_feature, zorder=-1)
        ax_map.set_extent([-113, -105, 35.5, 43.5], crs=crs.PlateCarree())

        for mk_type, style in mk_styles.items():
            subset = reservoirs_mktrend[mk_type][reservoirs_mktrend[mk_type].index.isin(res_df.columns)]
            if len(subset) == 0:
                continue

            ax_map.scatter(subset.LON, subset.LAT, transform=crs.PlateCarree(), zorder=1, **style)

            for _, row in subset.iterrows():
                offset = label_offsets.get(row['name'], dict(xytext=(6, 6), ha='center'))
                ax_map.annotate(row['name'], (row.LON, row.LAT), textcoords="offset points", fontsize=8, color='black', **offset)

    fig.tight_layout()
    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = 'reservoirs_data_overtime'
        fig.savefig(os.path.join(fig_dir, f'{save_name}.jpeg'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, f'{save_name}.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")
    plt.show()
    return fig,res_partitions, reservoirs_mktrend, label_offsets, drought_periods, mk_styles, row_config, res_names

    
## WT IN JULY
def plot_var_month_ann(var,month, y_label, sorted_var_month, sorted_var_ann,fig_dir,save=False ):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), gridspec_kw={'hspace': 0.35})
    fontsize = 14

    ax[0].set_ylabel(y_label, fontsize=fontsize + 1)
    ax[0].set_title(f"{month} Median {var}", fontsize=fontsize + 2)
    ax[1].set_ylabel(y_label, fontsize=fontsize + 1)
    ax[1].set_title(f"Annual Median {var}", fontsize=fontsize + 2)

    years = sorted_var_month.index

    for col in sorted_var_month.columns:
        ax[0].plot(years, sorted_var_month[col], label=col, linewidth=2)

    for col in sorted_var_ann.columns:
        ax[1].plot(years, sorted_var_ann[col], label=col, linewidth=2)

    # Drought rectangles — dynamically sized to each axis ylim
    drought_periods = [(2001, 2), (2012, 1), (2018, 1), (2020, 2)]

    for a in ax:
        a.grid(False)
        a.set_xticks(years)
        a.set_xticklabels(years, rotation=45, ha='right')
        for label in a.get_xticklabels() + a.get_yticklabels():
            label.set_fontsize(fontsize - 1)

        # Read ylim after data is plotted so height is correct
        ybot, ytop = a.get_ylim()
        rect_height = ytop - ybot
        for start, width in drought_periods:
            a.add_patch(mpl.patches.Rectangle(
                (start, ybot), width, rect_height,
                facecolor='pink', alpha=0.3, zorder=0
            ))

    handles, labels = ax[0].get_legend_handles_labels()

    ax[0].get_legend().remove() if ax[0].get_legend() else None

    # Add a single figure-level legend to the right
    fig.legend( handles, labels, loc='center left',
        bbox_to_anchor=(0.92, 0.5),   # right of figure, vertically centered
        prop={'size': 12}, ncols=1,frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.91, 1])   # leave room for legend on right
    
    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = var+'_'+month+'_allsites_timeseries'
        fig.savefig(os.path.join(fig_dir, f'{save_name}.jpeg'),
                    format='jpeg', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, f'{save_name}.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig

def plot_ann_distribution_for_pos_neg_RelChange(years, sites_by_rc, var_names, fig_dir=None, save=False):
    n_pos = {var: len(sites_by_rc[var]['pos']) for var in var_names}
    n_neg = {var: len(sites_by_rc[var]['neg']) for var in var_names}

    fig, ax = plt.subplots(3, 1, figsize=(6, 15))

    ax[0].hist(years['RDC'][sites_by_rc['RDC']['neg']].mean(), bins=10, alpha=0.5,  label=f'Sites with drought declines (n={n_neg["RDC"]})',  color='red')
    ax[0].hist(years['RDC'][sites_by_rc['RDC']['pos']].mean(), bins=10, alpha=0.75, label=f'Sites with drought increases (n={n_pos["RDC"]})', color='blue')
    ax[0].set_xlabel(r'Annual Average Q ($m^{3}.s^{-1}.km^{-2}$)')
    ax[0].set_ylabel('Number of Sites')
    ax[0].set_title('a) Annual Average Q (1998-2022)')
    ax[0].legend()

    ax[1].hist(years['WT'][sites_by_rc['WT']['pos']].mean(), bins=8, alpha=0.9, label=f'Sites with drought increases (n={n_pos["WT"]})', color='purple')
    ax[1].hist(years['WT'][sites_by_rc['WT']['neg']].mean(), bins=8, alpha=0.9, label=f'Sites with drought declines (n={n_neg["WT"]})',  color='orange')
    ax[1].set_xlabel(r'Annual Average WT ($^\circ$C)')
    ax[1].set_ylabel('Number of Sites')
    ax[1].set_title('b) Annual Average WT (1998-2022)')
    ax[1].legend()

    ax[2].hist(years['SC'][sites_by_rc['SC']['pos']].mean(), bins=7, alpha=0.9, label=f'Sites with drought increases (n={n_pos["SC"]})', color='purple')
    ax[2].hist(years['SC'][sites_by_rc['SC']['neg']].mean(), bins=7, alpha=0.8, label=f'Sites with drought declines (n={n_neg["SC"]})',  color='orange')
    ax[2].set_xlabel(r'Annual Average SC ($\mu$S.$cm^{-1}$)')
    ax[2].set_ylabel('Number of Sites')
    ax[2].set_title('c) Annual Average SC (1998-2022)')
    ax[2].legend()

    fig.tight_layout()
    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = 'All3var_pos_neg_differences_dist'
        fig.savefig(os.path.join(fig_dir, save_name+ '.jpeg'), format='jpeg', dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, save_name+ '.svg'), format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")
        
    plt.show()
    return fig,ax

# Used to plot the graphical abstract
def plot_avg_relchange(var_names, dr_names, mon_MED_relchange, MED_relchange_map,
                       wmon_names, var_rel_names, upper_colorado_river_boundary_dir, var_colors=None,
                       fig_dir=None, save=False):
    """
    Plot average relative change across all drought episodes for Q, WT, and SC
    in a single panel.

    Parameters:
    var_names (list): Hydrological variable names e.g. ['RDC', 'WT', 'SC']
    dr_names (list): Drought episode names
    mon_MED_relchange (dict): {var: {dr: DataFrame}} monthly relative change vs reference
    wmon_names (list): Water year month labels e.g. ['Oct', 'Nov', ...]
    var_rel_names (dict): {var: short_name} for legend labels
    var_colors (dict): {var: color} optional colors per variable
    fig_dir (str): Directory to save figures (required if save=True)
    save (bool): Whether to save the figure to disk

    Returns:
    fig, ax
    """
    fontsize = 16

    var_markers = {var: c for var, c in zip(var_names, ['^', '*', 'o'])}
    line_styles = {var: c for var, c in zip(var_names, ['-', '-.', ':'])}
    if var_colors is None:
        var_colors = {var: c for var, c in zip(var_names, ['steelblue', 'tomato', 'goldenrod'])}

    fig, ax = plt.subplots(figsize=(16, 9))

    projection = crs.AlbersEqualArea(central_latitude=39.5, central_longitude=-98.35)

    #left, bottom, width, height = [0.07, 0.573, 0.6, 0.41] #map on top 0.1, 0.09
    left, bottom, width, height = [0.04, 0.05, 0.3, 0.32] #map on bottom
    ax2 = fig.add_axes([left, bottom, width, height],projection=projection)
    
    shape_feature = basin_boundary(upper_colorado_river_boundary_dir)
    extent=(-113, -105, 35.5, 43.5)

    # Base map 
    ax2.add_feature(cfeature.COASTLINE, linestyle=':', edgecolor='grey', linewidth=1)
    ax2.add_feature(cfeature.STATES,    linestyle=':', edgecolor='grey', linewidth=1)
    ax2.add_feature(cfeature.RIVERS)
    ax2.set_extent(list(extent), crs=crs.PlateCarree())
    ax2.add_feature(shape_feature, zorder=-1)

    marker_alpha   = 0.8
    edge_linewidth = 0.3
    for var in var_names:
        data   = MED_relchange_map[var]
        marker = var_markers[var]

        ax2.scatter( x=data.LON, y=data.LAT, color=var_colors[var], edgecolors=var_colors[var],s=45, alpha=0.8,marker=marker, linewidth=edge_linewidth,
                transform=crs.PlateCarree(), zorder=2 )

    ### does the shading by dr episode
    for var in var_names:
        drought_medians = []
        for dr in dr_names:
            data = mon_MED_relchange[var][dr]
            drought_medians.append(data.median(axis=1))

        drought_medians_df = pd.concat(drought_medians, axis=1)
        avg_relchange = drought_medians_df.mean(axis=1)
        lower_bound   = drought_medians_df.min(axis=1)
        upper_bound   = drought_medians_df.max(axis=1)

        # Shading between min and max across droughts
        ax.fill_between(wmon_names, lower_bound, upper_bound,color=var_colors[var], alpha=0.2)

        ax.plot(wmon_names, avg_relchange,
            color=var_colors[var], marker=var_markers[var], markersize=15,
            linestyle=line_styles[var], label=var_rel_names[var], linewidth=2)

    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1)
    fig.text(0.09, 0.8, 'Drought Impacts in Upper Colorado River Basin: \nMedian Relative Changes During \n2001-02, 2012, 2018 and 2020-21 Drought Events', 
             fontsize=fontsize+10,ha='left')
    ax.set_ylabel('Relative Change (%)', fontsize=fontsize)
    ax.legend(fontsize=fontsize+2, loc='upper right')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fontsize)

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = 'graphical_abstract'
        fig.savefig(os.path.join(fig_dir, f'{save_name}.jpeg'),
                    dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, f'{save_name}.svg'),
                    format='svg', transparent=True, dpi=300, bbox_inches='tight')
        print(f"Saved figures to {fig_dir}")

    plt.show()
    return fig, ax, ax2


## Bar chart of years to recovery - x-axis will be annotated in Canva
def plot_recovery_years(recovery, var_names, dr_names, fig_dir = None, save= False):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    bins = np.linspace(1, 25, num=25)
    x = np.arange(len(bins) - 1)
    bar_width = 0.5

    ylims = {
        var_names[0]: (0, 125),
        var_names[1]: (0, 26),
        var_names[2]: (0, 13),
    }

    for i, var in enumerate(var_names):
    
        for j, dr in enumerate(dr_names[:-1]):
            ax = axes[i, j]
            data_pos_recov  = recovery[var][dr]['pos']['recyears_nononrec']
            data_neg_recov  = recovery[var][dr]['neg']['recyears_nononrec']
            data_pos_norecov  = recovery[var][dr]['pos']['notrecov_nomissing']
            data_neg_norecov  = recovery[var][dr]['neg']['notrecov_nomissing']
        
            pos_recov_counts, _ = np.histogram(data_pos_recov, bins=bins)
            neg_recov_counts, _ = np.histogram(data_neg_recov, bins=bins)
            notrecov_pos_counts, _ = np.histogram(data_pos_norecov, bins=bins)
            notrecov_neg_counts, _ = np.histogram(data_neg_norecov, bins=bins)

            ax.bar(x, pos_recov_counts, width=bar_width, alpha=0.75,
               label=f'Sites with drought increases (n={len(data_pos_recov)} )')
            ax.bar(x, neg_recov_counts, width=bar_width, bottom=pos_recov_counts, alpha=0.5,
               label=f'Sites with drought declines (n={len(data_neg_recov)} )')
            ax.bar(x, notrecov_pos_counts, width=bar_width,
               bottom=pos_recov_counts + neg_recov_counts, alpha=0.5,
               label=f'Non-return increases (n={len(data_pos_norecov)} )')
            ax.bar(x, notrecov_neg_counts, width=bar_width,
               bottom=pos_recov_counts + neg_recov_counts + notrecov_pos_counts, alpha=0.5,
               label=f'Non-return declines (n={len(data_neg_norecov)} )')

            label_var = 'Q' if var == var_names[0] else var
            ax.set_title(f'{label_var} Sites Years to Recovery Post-Drought {dr}')
            ax.set_xlabel('Years to Recovery')
            ax.set_ylabel('Number of Sites')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{int(edge)}' for edge in bins[:-1]], rotation=45)
            ax.set_xlim(-1, 25)
            ax.set_ylim(*ylims[var])
            ax.legend()

    fig.tight_layout()

    if save:
        if fig_dir is None:
            raise ValueError("fig_dir must be provided when save=True")
        save_name = 'All3_yearstorecoveryhist'
        fig.savefig(os.path.join(fig_dir, save_name+'.jpeg'), dpi=300)
        fig.savefig(os.path.join(fig_dir, save_name+'.svg'),
                    format='svg', transparent=True, dpi=300)
        print(f"Saved figures to {fig_dir}")
    plt.show()
    return fig, axes