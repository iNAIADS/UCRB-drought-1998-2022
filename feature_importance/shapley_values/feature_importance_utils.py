from copy import copy
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.inspection import permutation_importance

from scipy.stats import spearmanr

import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_mse_r2_training(mse_train, mse_test, r2_train, r2_test):
    """
    Plot training and test set training iterations
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.arange(len(mse_train[0])) + 1, np.mean(mse_train, axis=0), "b-", label="Training Set Deviance", lw=3) #reg.train_score_
    ax[0].plot(np.arange(len(mse_test[0])) + 1, np.mean(mse_test, axis=0), "r-", label="Test Set Deviance", lw=3)
    for i in range(mse_train.shape[0]):
        ax[0].plot(np.arange(len(mse_train[i])) + 1, mse_train[i], "b-", alpha=0.3)
        ax[0].plot(np.arange(len(mse_test[i])) + 1, mse_test[i], "r-", alpha=0.3)
            
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Boosting Iterations")
    ax[0].set_ylabel("RMSE")

    ax[1].plot(np.arange(len(r2_train[0])) + 1, np.mean(r2_train, axis=0), "b-", label="Training Set Deviance", lw=3) #reg.train_score_
    ax[1].plot(np.arange(len(r2_test[0])) + 1, np.mean(r2_test, axis=0), "r-", label="Test Set Deviance", lw=3)
    for i in range(r2_train.shape[0]):
        ax[1].plot(np.arange(len(r2_train[i])) + 1, r2_train[i], "b-", alpha=0.3)
        ax[1].plot(np.arange(len(r2_test[i])) + 1, r2_test[i], "r-", alpha=0.3)
            
    ax[1].legend(loc="lower right")
    ax[1].set_xlabel("Boosting Iterations")
    ax[1].set_ylabel("R^2")
    ax[1].set_ylim([-1,1])
    
    plt.tight_layout()
    plt.show()

def do_hyperparams_gridsearch(
    X,
    y,
    param_grid = {
        'learning_rate': [0.01,0.1, 0.3],
        'max_depth': [2, 3, 5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0,1,10],
        'reg_lambda': [0,1,10]
    }
    ):

    grid = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=42),
        verbose=1,
        n_jobs=4,
        return_train_score=True,
    )
    grid.fit(X, y)
    return grid.best_params_


def plot_importance_barchart(feat_imp_mean, feat_imp_se, spear_corr, filename, labels=None, suffix='', max_num_bars=40, save_values_csv=True):
    color_list = ["red", "white", "blue"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", color_list)
    norm_color = copy(spear_corr)
    norm_color = (norm_color+1.)/2.
    sorted_idx = np.argsort(feat_imp_mean)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(20,12), facecolor='white')
    ax = fig.add_axes((0.35, 0.1, 0.45, .85))
    
    bars_to_plot = max_num_bars
    ax.barh(pos[-bars_to_plot:], feat_imp_mean[sorted_idx][-bars_to_plot:], align="center",
            xerr=feat_imp_se[sorted_idx][-bars_to_plot:],
            color=cmap(norm_color[sorted_idx][-bars_to_plot:]),
            edgecolor='k'
           )
    if labels is not None:
        ax.set_yticks(pos[-bars_to_plot:], labels[sorted_idx][-bars_to_plot:])
    ax.set_xlabel("Importance", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=20)
    sm = plt.cm.ScalarMappable(cmap=mcolors.LinearSegmentedColormap.from_list("", color_list),
                                norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1.))
    sm.set_array([])
    cax = fig.add_axes([.83, 0.2, 0.03, 0.6])
    bar = plt.colorbar(sm, cax=cax, shrink=1.)
    cax.set_title("Spearman correlation", y=0.45, x=3.5, pad=-104, rotation=90, fontsize=20)
    bar.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig("best_RFE_traits_importance_new_%s_%s" % (filename[:-4], suffix), dpi=300)
    plt.show()

    if save_values_csv==True:
        pd.DataFrame(index=labels[sorted_idx][-bars_to_plot:], data=np.array([feat_imp_mean[sorted_idx][-bars_to_plot:]]).T, columns=["Importance"]).to_csv("best_RFE_traits_importance_new_%s_%s.csv" % (filename[:-4], suffix))

def load_dataset(filename = 'drought_withnans_relchange_1-13-2025_all_nanmean_noOutliers.csv'):
    df_traits_WQ = pd.read_csv(filename)
    
    traits = df_traits_WQ.columns.values[:-1]
    WQ = df_traits_WQ.columns.values[-1]
    
    X = df_traits_WQ[traits].values
    y = df_traits_WQ[WQ].values

    idxs_same_values = np.array([len(np.unique(X[:,i]))==1 for i in range(X.shape[1])])
    traits_same_values = traits[idxs_same_values]
    print("Removed traits because unique values:", traits_same_values)
    
    traits = traits[~idxs_same_values]
    X = X[:, ~idxs_same_values]
    
    # Standardize features
    X = (X-X.mean(axis=0))/X.std(axis=0)
    y = (y-np.mean(y))/np.std(y)
    
    return X, y, traits

def plot_RFE(x_RFE, rmse_ave_RFE, r2_ave_RFE, idx_traits_best_RFE):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_xlabel("Number of features selected")
    ax[0].set_ylabel("RMSE")
    ax[0].plot(x_RFE, rmse_ave_RFE)
    
    ax[1].set_xlabel("Number of features selected")
    ax[1].set_ylabel("R^2")
    ax[1].plot(x_RFE, r2_ave_RFE)
    
    ax[0].plot(x_RFE[idx_traits_best_RFE], rmse_ave_RFE[idx_traits_best_RFE], 'o', color='red')
    ax[1].plot(x_RFE[idx_traits_best_RFE], r2_ave_RFE[idx_traits_best_RFE], 'o', color='red')
    
    plt.show()

