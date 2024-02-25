from pathlib import Path
from itertools import product

import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats
from ruamel.yaml import YAML

from neuralhydrology.datasetzoo.camelsus import CamelsUS
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils.utils import load_scaler

# This is the code for the synthetic experiment

# settings:
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default=0.0)
args = vars(parser.parse_args())
yaml = YAML()
cfg = yaml.load(Path(args["cfg"]))

epxeriment_name = cfg["experiment_name"]
PATH_NH_CFG = Path(cfg["path_nh_cfg"])
PATH_NH_SCALER = Path(cfg["path_nh_scaler"])
PATH_OUT = Path(cfg["path_out"])
BASIN_ID = cfg["basin_id"]
DATA_DIVISION = cfg["data_division"]
OPTIM_CRITERION = cfg["optim_criterion"]
metric_optim_below = np.float32(cfg["metricbelow"])
N_ITER = cfg["n_iter"]

RUN_DEVICE = torch.device(cfg["cuda"])

GRID_LENGTHS = (20, 20)  # number of runs will be x_w*x_nse
eps = 0.0001
bounds_weights = [0.1, 0.9]
bounds_metric = [0.1, 0.9]
###################################################################

# create folder to save stuff
path_save_end = PATH_OUT / f"{OPTIM_CRITERION}"
Path(path_save_end).mkdir(parents=True, exist_ok=True)

# prepare grid:
w_samples = np.linspace(bounds_weights[0], bounds_weights[1], GRID_LENGTHS[0])
metric_optim_above_samples = np.linspace(
    bounds_metric[0], bounds_metric[1], GRID_LENGTHS[1]
)

# create list of combinations
combinations = list(product(w_samples, metric_optim_above_samples))
temp = list(zip(*combinations))
temp[0] = np.array(temp[0])
temp[1] = np.array(temp[1])
# collect grid in dictionary and add entry for {OPTIM_CRITERION}_all:
combi_dict = dict(zip(["w", f"{OPTIM_CRITERION}_above"], temp))
combi_dict[f"{OPTIM_CRITERION}_all"] = (
    0.0 * combi_dict[f"{OPTIM_CRITERION}_above"]
)  # HACK. This not nice, but it works

scaler = load_scaler(PATH_NH_SCALER)
cfg = Config(PATH_NH_CFG)
Dataset = CamelsUS
ds = Dataset(cfg=cfg, is_train=False, period="train", scaler=scaler)


# define functions: -----------------------------------------------
def create_data(basin_id, w):
    basin_data = ds._load_basin_data(BASIN_ID)
    obs_base = basin_data["QObs(mm/d)"].values

    if DATA_DIVISION.lower() == "minmax":
        threshold = (1.0 - w) * np.nanmin(obs_base) + w * np.nanmax(obs_base)
    elif DATA_DIVISION.lower() == "quantiles":
        threshold = np.nanquantile(obs_base, w)
    else:
        raise ValueError(f"data division ({DATA_DIVISION}) does not exist")

    normalizer = {"std": np.nanstd(obs_base) + eps}  # 'mean': np.nanmean(obs_base)

    obs = obs_base / normalizer["std"]
    noise_signal = torch.from_numpy(obs + 0.05 * np.random.normal(size=obs.shape))
    obs = torch.from_numpy(obs)
    obs.to(RUN_DEVICE)

    sim = noise_signal
    sim.requires_grad_(True)
    sim.to(RUN_DEVICE)

    return obs, sim, normalizer, threshold


# metrics:
def eval_cor(obs, sim):
    idx1 = ~np.isnan(obs)
    idx2 = ~np.isnan(sim)
    idx = idx1 & idx2
    #
    obs = obs[idx]
    sim = sim[idx]

    if len(obs) < 2:
        return np.nan

    r, _ = stats.pearsonr(obs, sim)

    return float(r)


def eval_kge(obs, sim):
    idx1 = ~np.isnan(obs)
    idx2 = ~np.isnan(sim)
    idx = idx1 & idx2
    #
    obs = obs[idx]
    sim = sim[idx]

    if len(obs) < 2:
        return np.nan

    pearsonr, _ = stats.pearsonr(obs, sim)
    alpha = sim.std() / (obs.std() + eps)
    beta = sim.mean() / (obs.mean() + eps)

    value = (pearsonr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2

    return 1 - np.sqrt(float(value))


def eval_lense(obs, sim, obs_ref):
    idx1 = ~np.isnan(obs)
    idx2 = ~np.isnan(sim)
    idx = idx1 & idx2
    #
    obs = obs[idx]
    sim = sim[idx]
    obs_ref = obs_ref[~np.isnan(obs_ref)]

    denominator = ((obs_ref - obs_ref.mean() + eps) ** 2).mean() + eps
    numerator = ((sim - obs) ** 2).mean()

    return 1 - numerator / denominator


def eval_nse(obs, sim):
    idx1 = ~np.isnan(obs)
    idx2 = ~np.isnan(sim)
    idx = idx1 & idx2
    #
    obs = obs[idx]
    sim = sim[idx]

    denominator = ((obs - obs.mean() + eps) ** 2).sum() + eps
    numerator = ((sim - obs) ** 2).sum()

    return 1 - numerator / denominator


def evaluate(obs, sim, obs_ref):
    if OPTIM_CRITERION.lower() == "nse":
        metric = eval_nse(obs, sim)
    elif OPTIM_CRITERION.lower() == "lense":
        metric = eval_lense(obs, sim, obs_ref)
    elif OPTIM_CRITERION.lower() == "kge":
        metric = eval_kge(obs, sim)
    elif OPTIM_CRITERION.lower() == "cor":
        metric = eval_cor(obs, sim)
    else:
        raise ValueError(
            f"Your evaluation critertion ({OPTIM_CRITERION}) does not exist"
        )

    return metric


# losses:
def loss_cor(obs, sim):
    idx = ~torch.isnan(obs)
    obs_sub = obs[idx]
    sim_sub = sim[idx]

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    pearson = cos(obs_sub - obs_sub.mean(), sim_sub - sim_sub.mean())
    return pearson


def loss_kge(obs, sim):
    idx = ~torch.isnan(obs)
    obs_sub = obs[idx]
    sim_sub = sim[idx]

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    pearsonr = cos(obs_sub - obs_sub.mean(), sim_sub - sim_sub.mean())
    alpha = torch.std(sim) / torch.std(obs)
    beta = torch.mean(sim) / torch.mean(sim)
    value = (pearsonr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    return 1 - torch.sqrt(value)


def loss_lense(obs, sim, obs_ref):
    idx = ~torch.isnan(obs)
    obs_sub = obs[idx]
    sim_sub = sim[idx]
    obs_ref_sub = obs_ref[~torch.isnan(obs_ref)]

    numerator = torch.mean((sim_sub - obs_sub) ** 2)
    denominator = torch.mean((obs_ref_sub - obs_ref_sub.mean()) ** 2) + eps
    return 1 - numerator / denominator


def loss_nse(obs, sim):
    idx = ~torch.isnan(obs)
    obs_sub = obs[idx]
    sim_sub = sim[idx]

    numerator = torch.sum((sim_sub - obs_sub) ** 2)
    denominator = torch.sum((obs_sub - obs_sub.mean()) ** 2) + eps
    return 1 - numerator / denominator


def compute_loss_metric(obs, sim, obs_ref=0):
    if OPTIM_CRITERION.lower() == "nse":
        loss = loss_nse(obs, sim)
    elif OPTIM_CRITERION.lower() == "lense":
        loss = loss_lense(obs, sim, obs_ref=obs_ref)
    elif OPTIM_CRITERION.lower() == "kge":
        loss = loss_kge(obs, sim)
    elif OPTIM_CRITERION.lower() == "cor":
        loss = loss_cor(obs, sim)
    else:
        raise ValueError(
            f"The optimization critertion ({OPTIM_CRITERION}) does not exist"
        )
    return loss


def loss_composite(
    y, y_hat, normalizer, threshold, optim_below=0.9, optim_above=0.7, obs_ref=0
):
    y = y * normalizer["std"]
    y_hat = y_hat * normalizer["std"]

    obs_below = y[y <= threshold]
    sim_below = y_hat[y <= threshold]
    metric_below = compute_loss_metric(obs_below, sim_below, obs_ref)
    loss_below = torch.abs(optim_below - metric_below)

    obs_above = y[y > threshold]
    sim_above = y_hat[y > threshold]
    metric_above = compute_loss_metric(obs_above, sim_above, obs_ref)
    loss_above = torch.abs(optim_above - metric_above)

    return loss_below + loss_above


def optimize_for_loss(
    obs,
    sim,
    normalizer,
    optimizer,
    threshold,
    obs_ref,
    optim_below=0.9,
    optim_above=0.7,
    n_iter=50000,
):
    clamp_max = 2 * torch.max(sim[~torch.isnan(sim)])
    clamp_max = clamp_max.detach().cpu().numpy()
    for n in range(n_iter - 1):
        optimizer.zero_grad()
        sim_map = torch.clamp(sim, 0.0, float(clamp_max))  # bind sim
        loss = loss_composite(
            obs,
            sim_map,
            normalizer=normalizer,
            threshold=threshold,
            optim_below=optim_below,
            optim_above=optim_above,
            obs_ref=obs_ref,
        )
        loss.backward()
        optimizer.step()


# run:
for n in range(len(combinations)):
    w = combi_dict["w"][n]
    metric_optim_above = combi_dict[f"{OPTIM_CRITERION}_above"][n]
    obs, sim, normalizer, threshold = create_data(basin_id=BASIN_ID, w=w)
    if n == 0:
        obs_ref = obs.detach().clone()
    optimizer = torch.optim.SGD([sim], lr=1.0)

    optimize_for_loss(
        obs,
        sim,
        normalizer=normalizer,
        optimizer=optimizer,
        threshold=threshold,
        obs_ref=obs_ref,
        optim_below=metric_optim_below,
        optim_above=metric_optim_above,
    )

    obs_all = obs.detach().cpu().numpy()
    obs_all = obs_all * normalizer["std"]
    sim_all = sim.detach().cpu().numpy()
    sim_all = sim_all * normalizer["std"]

    metric_all = evaluate(obs=obs_all, sim=sim_all, obs_ref=obs_ref.numpy())
    combi_dict[f"{OPTIM_CRITERION}_all"][n] = metric_all

    print(
        f"({n+1}/{len(combinations)})> low:{metric_optim_below:.3f}, all:{metric_all:.3f}, high:{metric_optim_above:.3f}"
    )


# save:
results = pd.DataFrame.from_dict(combi_dict)
results[f"{OPTIM_CRITERION}_below"] = metric_optim_below
results.to_csv(
    path_save_end
    / f"results-{OPTIM_CRITERION}-basin#{BASIN_ID}-below{metric_optim_below:.2f}-{epxeriment_name}.csv"
)
