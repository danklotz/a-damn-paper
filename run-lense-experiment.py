import argparse

import netCDF4
import numpy as np
import pandas as pd

from pathlib import Path
from ruamel.yaml import YAML
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# This is the code that forms the basis of the comparative analysis
#
# The code itself does much more than required for the experiment,
# however this makes it possible to experiment with it.

# load cfg:
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default=0.0)
args = vars(parser.parse_args())
yaml = YAML()
cfg = yaml.load(Path(args["cfg"]))

# set options:
data_dir = Path(cfg["data_dir"])
ref_model_dir = Path(cfg["ref_model_dir"])
window_size = cfg["window_size"]
window_point_of_interest = cfg["window_point_of_interest"]
model_id = cfg["model_id"]
#
n_neighbors = [10, 20, 30, 40, 50, 100, 200, 400, 800]
period_idx = 0
period_starts = ["1989-10-01"]
period_ends = ["1999-09-30"]

# load data:
ds = netCDF4.Dataset(ref_model_dir)
model_list = list(ds.variables)[2:]
model_list.remove("date")
basins = np.array(ds["basin"])


# setup experiment:
def metric_nse(obs, sim, ref_mean, eps=0.0):
    idx1 = ~np.isnan(obs)
    idx2 = ~np.isnan(sim)
    idx = idx1 & idx2
    #
    obs = obs[idx]
    sim = sim[idx]

    numerator = ((obs - sim) ** 2).sum()
    denominator = ((obs - ref_mean) ** 2).sum() + eps

    return 1 - numerator / denominator


def load_camels_us_discharge(data_dir: Path, basin: str) -> pd.Series:
    discharge_path = data_dir / "usgs_streamflow"
    file_path = list(discharge_path.glob(f"**/{basin}_streamflow_qc.txt"))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No file for Basin {basin} at {file_path}")

    col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    df = pd.read_csv(file_path, sep="\s+", header=None, names=col_names)
    df["date"] = pd.to_datetime(
        df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
        format="%Y/%m/%d",
    )
    df = df.set_index("date")

    # load area from forcings from daymet:
    forcing_path = data_dir / "basin_mean_forcing" / "daymet"
    file_path = list(forcing_path.glob(f"**/{basin}_*_forcing_leap.txt"))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No file for Basin {basin} at {file_path}")

    with open(file_path, "r") as fp:
        fp.readline()
        fp.readline()
        area = int(fp.readline())

    # normalize discharge from cubic feet per second to mm per day:
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs


# run lense experiment:
pc_collection = {
    "window_size": np.full(len(basins) * len(n_neighbors), window_size),
    "k_neighbors": np.repeat(n_neighbors, len(basins)),
    "model": np.repeat(model_id, len(basins) * len(n_neighbors)),
    "basin": np.tile(basins, len(n_neighbors) * 1),
    "pc (mean)": np.zeros(len(basins) * len(n_neighbors)),
}
pc_collection = pd.DataFrame(pc_collection)
validation_dates = netCDF4.num2date(
    ds.variables["date"], ds.variables["date"].units, only_use_cftime_datetimes=False
)

for basin_id in tqdm(basins):
    obs_all_pd = load_camels_us_discharge(data_dir, basin_id)
    obs_validation = obs_all_pd.loc[
        period_starts[period_idx] : period_ends[period_idx]
    ].values
    obs_validation_mean = np.mean(obs_validation)

    # divide the observations into overlapping segments of size 'window_size':
    all_obs_segments = [
        obs_validation[i : i + window_size]
        for i in range(len(obs_validation) - window_size + 1)
    ]

    for k_neighbors in n_neighbors:
        # fit NN:
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(all_obs_segments)

        # evaluate per model:
        basin_selection = basins == basin_id

        # get simulations:
        sim_all_pd = pd.DataFrame(
            {"sim": np.array(ds[model_id][basin_selection])[0]},
            index=pd.to_datetime(validation_dates),
        )
        sim_validation = sim_all_pd.loc[
            period_starts[period_idx] : period_ends[period_idx]
        ].values[:, 0]

        # divide the simulation into overlapping segments of size 'window_size':
        all_sim_segments = [
            sim_validation[i : i + window_size]
            for i in range(len(sim_validation) - window_size + 1)
        ]

        # compute metrics:
        pc_per_timestep = []
        cumulative_weight = 0
        for segment_idx in range(len(all_obs_segments)):
            obs_segment = all_obs_segments[segment_idx]
            sim_segment = all_sim_segments[segment_idx]

            indices = knn.kneighbors([obs_segment], return_distance=False)

            weight = 1 / (
                np.var(
                    [all_obs_segments[i][window_point_of_interest] for i in indices[0]]
                )
                + 0.000001
            )
            cumulative_weight += weight
            error_term = (
                obs_segment[window_point_of_interest]
                - sim_segment[window_point_of_interest]
            ) ** 2
            pc_per_timestep.append(error_term * weight)
        pc_per_timestep = np.array(pc_per_timestep)

        row_mask = (pc_collection["k_neighbors"] == k_neighbors) & pc_collection[
            "basin"
        ].str.match(basin_id)
        pc_collection.loc[row_mask, "pc (mean)"] = (
            1 - np.sum(pc_per_timestep) / cumulative_weight
        )

# save:
pc_collection.to_csv(Path(cfg["output"]))
