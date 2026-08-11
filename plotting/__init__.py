# plotting/__init__.py

"""
Analysis plotting utilities.
"""

from .plotting import (
    setup_plotting,
    plot_data_availability,    
    plotting_spei,
    plot_dr_ref_scatter,
    plot_combined_monthly_analysis,
    plot_monthly_relchange_boxplots,
    plot_mk_panel,
    plot_combined_mk_trends,
    get_facecolors,
    add_cbar_triangles,
    basin_boundary,
    _scatter_matched_edges,
    _scatter_invisible,
    plot_relchange_map,
    plot_colocation_relchange,
    plot_colocation_seasonal_relchange,
    plot_relchange_percentile_maps,
    plot_pq_scatter_ex_sites,
    partition_reservoirs,
    prep_reservoir_line_map,
    plot_reservoir_line_map,
    plot_var_month_ann,
    plot_ann_distribution_for_pos_neg_RelChange,
    plot_avg_relchange,
    plot_recovery_years
)

__all__ = [
    "setup_plotting",
    "plot_data_availability",
    "plotting_spei",
    "plot_dr_ref_scatter",
    "plot_combined_monthly_analysis",
    "plot_monthly_relchange_boxplots",
    "plot_mk_panel",
    "plot_combined_mk_trends",
    "get_facecolors",
    "add_cbar_triangles",
    "basin_boundary",
    "_scatter_matched_edges",
    "_scatter_invisible",
    "plot_relchange_map",
    "plot_colocation_relchange",
    "plot_colocation_seasonal_relchange",
    "plot_relchange_percentile_maps",
    "plot_pq_scatter_ex_sites",
    "partition_reservoirs",
    "prep_reservoir_line_map",
    "plot_reservoir_line_map",
    "plot_var_month_ann",
    "plot_ann_distribution_for_pos_neg_RelChange",
    "plot_avg_relchange",
    "plot_recovery_years"
]