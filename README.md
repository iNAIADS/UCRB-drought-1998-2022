# UCRB-drought
This repository stores code for the event-based drought analysis on the Upper Colorado River Basin (UCRB) described in Nagamoto et al, 2025 in prep. 

## Contents

```
├── calculations  #code used in UCRB_Drought_Workflow.ipynb
│   ├── calculations.py
│   ├── __init__.py
│   └── setup.py
├── climatic_variables  #code to access variables used in UCRB_Drought_Workflow.ipynb
│   ├── catchments_climatic_spatial_aggregation.ipynb
│   ├── Readme.txt
│   └── requirements.txt
├── feature_importance  #code using data calculated in UCRB_Drought_Workflow.ipynb
│   ├── permutation_importance
│   │   ├── Readme.md
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_aggr_permutation.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_aggr_permutation.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_permutation.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_permutation.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_aggr_permutation.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_aggr_permutation.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_permutation.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_permutation.png
│   │   ├── communities_attr.json
│   │   ├── dict_trait_names.json
│   │   ├── drought_withnans_relchange_1-13-2025_all_nanmedian_new.csv
│   │   ├── drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers.csv
│   │   ├── feature_importance_utils.py
│   │   ├── results_drought_withnans_relchange_1-13-2025_all_nanmedian_new_permutation.json
│   │   ├── results_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_permutation.json
│   │   ├── traits_categories_labels.json
│   │   └── xgb_optimizer_for_regression.ipynb
│   ├── shapley_values
│   │   ├── Readme.md
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_aggr_shapley.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_aggr_shapley.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_shapley.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_new_shapley.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_aggr_shapley.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_aggr_shapley.png
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_shapley.csv
│   │   ├── best_RFE_traits_importance_new_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_shapley.png
│   │   ├── communities_attr.json
│   │   ├── dict_trait_names.json
│   │   ├── drought_withnans_relchange_1-13-2025_all_nanmedian_new.csv
│   │   ├── drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers.csv
│   │   ├── feature_importance_utils.py
│   │   ├── results_drought_withnans_relchange_1-13-2025_all_nanmedian_new_permutation.json
│   │   ├── results_drought_withnans_relchange_1-13-2025_all_nanmedian_noOutliers_permutation.json
│   │   ├── traits_categories_labels.json
│   │   └── xgb_optimizer_for_regression.ipynb  
│   ├── Readme.md
│   ├── __init__.py
│   ├── chatGPT_prompt_categories.txt
│   ├── feature_importance_utils.py
│   └── setup.py
├── preprocessing  #code used in UCRB_Drought_Workflow.ipynb
│   ├── preprocessing.py
│   ├── __init__.py
│   └── setup.py
├── pyeto  #code used in UCRB_Drought_Workflow.ipynb from https://github.com/woodcrafty/PyETo/tree/master
│   ├── __init__.py
│   ├── _check.py
│   ├── convert.py
│   ├── fao.py
│   ├── setup.py
│   └── thornthwaite.py
├── README.md
├── requirements_ucrb-drought.yml
└── UCRB_Drought_Workflow.ipynb

```

## User Information
The workflow of this uses prepackaged data located on ESS-DIVE (doi:10.15485/2551894) that includes an INPUTS folder with physical catchment attributes, meteorologic data, National Land Cover data, boundary information for the UCRB, and daily observations of streamflow, water temperature, and specific conductance at catchments across the UCRB, in addition to calculated water temperature data from [stream-temperature-ml-ensembles](https://github.com/iNAIADS/stream-temperature-ml-ensembles/tree/main).

When running the UCRB_Drought_Workflow.ipynb, intermediate data and outputs will be saved to an OUTPUTS folder located in the directory where this code is used locally.

### Datasets
Start by downloading the prepackaged data and code located on ESS-DIVE (doi:10.15485/2551894). This contains folders for all of the input data, as well as the outputs and figures generated. The code is also contained there, following the same folder structure as presented here. It is structured so that you do not need to move anything and can run the workflow as indicated below, but you may wish to move this folder to where you want to run it. 

### Packages
In this project we use micromamba for managing python environment. Download either Anaconda or Mamba from here:
* [Anaconda](https://www.anaconda.com/products/individual) 
* [Mamba](https://github.com/mamba-org/mamba) 

After cloning the repository, `cd` into the repository (`cd path/to/ucrb-drought`) and create an environment using the requirements_ucrb-drought.yml file.

In the same command prompt / terminal opened to the `ucrb-drought` directory, run the following code:

```
mamba create -f requirements_ucrb-drought.yml
```

This will create a conda environment called `ucrb-drought` and download the required packages. Activate this conda environment and create a kernel in Jupyter Lab to run the Notebooks

```
mamba activate ucrb-drought 

python -m ipykernel install --user --name=ucrb-drought
```

Now, you can open the notebooks in Jupyter Lab.
```
jupyter lab
```

## Directions to run the code used in paper
You must begin by downloading the prepackaged data and code located on ESS-DIVE (doi:10.15485/2551894). While you can clone this repository using the different clone options, since the code is also contained in the ESS-DIVE file, you may use that. Note that this repository will be the most up to date place for code revisions. After setting up your data and code folders where you want them to be (see `Datasets`), create a virtual environment using the `Packages` section above. 

### MAIN WORKFLOW
After opening jupyter lab, open the main workflow file: `UCRB_Drought_Workflow.ipynb`. This notebook utilizes python code in `calculations`, `preprocessing`, and `pyeto`, the latter of which comes from [PyETo](https://github.com/woodcrafty/PyETo/tree/master). In `UCRB_Drought_Workflow.ipynb`, streamflow, water temperature, specific conductance, and meteorologic data is preprocessed and cleaned, with these intermediate outputs saved to OUTPUTS. The meteorologic data is collected using Google Earth Engine in `climatic_variables/catchments_climatic_spatial_aggregation.ipynb`, but the raw data is also ready in the INPUTS folder on ESS-DIVE. 

Run all cells in the `UCRB_Drought_Workflow.ipynb` notebook to complete the analysis. The final cells at the end of the notebook produce visualizations of the data that were used in the associated paper.

### FEATURE IMPORTANCE
After obtaining the Relative Change percentages using `UCRB_Drought_Workflow.ipynb`, the streamflow values along with physical catchment attributes from [GAGESII](https://pubs.usgs.gov/publication/70046617) are used in `feature_importance/permutation_importance/xgb_optimizer_for_regression.ipynb` and `feature_importance/shapley_values/xgb_optimizer_for_regression.ipynb` to investigate with attributes may be predictive of catchment streamflow vulnerability to drought. These notebooks, while separate, use the same virtual environment, and utilize python code in `feature_importance/feature_importance_utils.py`.


## References

[Nagamoto E ; Ciulla F ; Ombadi M ; Willard J ; Carroll R ; Varadharajan C (2025): Dataset: "Widespread Drought-driven Declines in Streamflows and Water quality in the Upper Colorado River Basin (1998-2022)". iNAIADS, ESS-DIVE repository. Dataset. doi:10.15485/2551894 accessed via https://data.ess-dive.lbl.gov/datasets/doi:10.15485/2551894 on 2025-04-25](https://data.ess-dive.lbl.gov/view/doi%3A10.15485%2F2551894)

[Willard J ; Varadharajan C (2025): Dataset for Willard et al. (2025) "Machine Learning Ensembles Can Enhance Hydrologic Predictions and Uncertainty Quantification" iNAIADS, ESS-DIVE repository. Dataset. doi:10.15485/2448016](https://data.ess-dive.lbl.gov/view/doi%3A10.15485%2F2527393)

[Richards, M. PyETo. Library. Accessed via https://github.com/woodcrafty/PyETo/tree/master on 2025-05-22](https://github.com/woodcrafty/PyETo/tree/master)

[Falcone, J (2011). "GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow". USGS Water Mission Area NSDI Node. U.S. Geological Survey. Accessed via https://catalog.data.gov/dataset/gages-ii-geospatial-attributes-of-gages-for-evaluating-streamflow/resource/d3ebdf37-e18f-48c1-9105-9661fde648eb on  on 2025-05-22](https://catalog.data.gov/dataset/gages-ii-geospatial-attributes-of-gages-for-evaluating-streamflow/resource/d3ebdf37-e18f-48c1-9105-9661fde648eb)

Nagamoto et al (in prep): Widespread Drought-driven Declines in Streamflows and Water quality in the Upper Colorado River Basin (1998-2022).


## Funding
This work is supported by the iNAIADS Early Career Research Program award and the Watershed Function Science Focus Area funded by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research under the Berkeley Lab Contract Number DE-AC02-05CH11231. This work was also supported in part by the U.S. Department of Energy, Office of Science, Office of Workforce Development for Teachers and Scientists (WDTS) under the Science Undergraduate Laboratory Internship (SULI) program. This research also used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility located at Lawrence Berkeley National Laboratory, operated under Contract No. DE-AC02-05CH11231.

## License
This code is available with a CCBy4.0 license published in this dataset: [Nagamoto E ; Ciulla F ; Ombadi M ; Willard J ; Carroll R ; Varadharajan C (2025): Dataset: "Regional declines in stream flows and water quality in the Upper Colorado River Basin due to meteorological droughts (1998-2022)". iNAIADS, ESS-DIVE repository. Dataset. doi:10.15485/2551894 accessed via https://data.ess-dive.lbl.gov/datasets/doi:10.15485/2551894 on 2025-04-25](https://data.ess-dive.lbl.gov/view/doi%3A10.15485%2F2551894). This dataset should be cited for any use of this code. 
