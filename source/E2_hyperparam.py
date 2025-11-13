import numpy as np

# this file is created to analyze the data for experiment E2
# it is fast debug code -> not clean
# the main metric of this experiment is average percentile
T_values = ["1", "5", "25", "50", "75", "100"]
T_hold_values = ["1", "5", "25", "50", "75", "100"]
eta_rels = ["01", "02", "03", "04", "05"]
eta_ups = ["03", "05", "07", "09", "11"]
h_rels = ["01", "02", "03", "04", "05"]
h_ups = ["04", "06", "08", "10", "12"]

for T in T_values:
    path = f"./runs/E2_hyperparams/T/T_{T}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    T_data = np.load(path)
    print(f"Mean (T = {T}): {np.round(np.mean(T_data), 2)}")

for T_hold in T_hold_values:
    path = f"./runs/E2_hyperparams/T_hold/T_hold_{T_hold}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    T_hold_data = np.load(path)
    print(f"Mean (T_hold = {T_hold}): {np.round(np.mean(T_hold_data), 2)}")

for eta_rel in eta_rels:
    path = f"./runs/E2_hyperparams/eta_rel/eta_rel_{eta_rel}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    eta_rel_data = np.load(path)
    print(f"Mean (eta_rel = {eta_rel}): {np.round(np.mean(eta_rel_data), 2)}")

for eta_up in eta_ups:
    path = f"./runs/E2_hyperparams/eta_up/eta_up_{eta_up}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    eta_up_data = np.load(path)
    print(f"Mean (eta_up = {eta_up}): {np.round(np.mean(eta_up_data), 2)}")

for h_rel in h_rels:
    path = f"./runs/E2_hyperparams/h_rel/h_rel_{h_rel}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    h_rel_data = np.load(path)
    print(f"Mean (h_rel = {h_rel}): {np.round(np.mean(h_rel_data), 2)}")

for h_up in h_ups:
    path = f"./runs/E2_hyperparams/h_up/h_up_{h_up}/simulation_results/gap_experiment_3_seed_7/simulation_data/percentile_level.npy"
    h_up_data = np.load(path)
    print(f"Mean (h_up = {h_up}): {np.round(np.mean(h_up_data), 2)}")