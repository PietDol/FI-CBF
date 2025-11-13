import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# file to make plots
# function to give safety margin and time combo
def safety_margin_helper(path):
    # path to sim data
    t = np.load(f"{path}/control_time.npy")
    safety_margin = np.load(f"{path}/safety_margin.npy")
    # take only dominant safety margin per timestep
    safety_margin = np.max(safety_margin, 1)
    return t, safety_margin


def plot_safety_margin_E1(gap_paths, cluttered_paths):
    # Embed fonts nicely in PDF
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )

    W = 7.16  # your full \textwidth in inches
    H = 0.47 * W  # good aspect for 2 stacked plots
    fig, ax = plt.subplots(2, 1, figsize=(W, H), sharex=True, constrained_layout=True)

    # Gap env
    for i, path in enumerate(gap_paths):
        t, s = safety_margin_helper(path)
        ax[0].plot(t, s, lw=1.0, label=f"R{i}")

    # Cluttered env
    for i, path in enumerate(cluttered_paths):
        t, s = safety_margin_helper(path)
        ax[1].plot(t, s, lw=1.0, label=f"R{i}")

    # Styling
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.set_axisbelow(True)
        a.set_ylabel(r"$M(\widehat{\mathbf{x}})$ [m]")
        a.margins(x=0)
        a.legend(loc="center right", framealpha=0.9)

    ax[1].set_xlabel(r"$t$ [s]")  # only bottom subplot gets xlabel
    ax[0].set_title(
        r"Safety margin $M(\widehat{\mathbf{x}})$ over time for Gap Environment"
    )
    ax[1].set_title(
        "Safety margin $M(\widehat{\mathbf{x}})$ over time for Cluttered Environment"
    )

    fig.savefig("E1_safety_margins.pdf", bbox_inches="tight", transparent=True)

def plot_safety_margin_E3(gap_paths):
    # Embed fonts nicely in PDF
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )

    W = 7.16  # your full \textwidth in inches
    H = 0.27 * W  # good aspect for 2 stacked plots
    fig, ax = plt.subplots(1, 1, figsize=(W, H), sharex=True, constrained_layout=True)
    values = [0.1, 0.15, 0.2]
    # Gap env
    for i, path in enumerate(gap_paths):
        t, s = safety_margin_helper(path)
        ax.plot(t, s, lw=1.0, label=rf"$\sigma_{{\max}}={values[i]}$")

    # Styling
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"$M(\widehat{\mathbf{x}})$ [m]")
    ax.margins(x=0)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_xlabel(r"$t$ [s]")  # only bottom subplot gets xlabel
    ax.set_title(
        r"Safety margin $M(\widehat{\mathbf{x}})$ over time for Gap Environment"
    )

    fig.savefig("E3_safety_margins.pdf", bbox_inches="tight", transparent=True)

def noise_helper(path):
    t = np.load(f"{path}/control_time.npy")
    noise_true = np.load(f"{path}/noise_true.npy")
    noise = np.load(f"{path}/noise.npy")
    return t, noise_true, noise


def plot_noise(gap_paths):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # gap env
    for i, path in enumerate(gap_paths):
        t, noise_true, noise = noise_helper(path)
        ax[i].plot(t, noise, label=f"Est. noise")
        ax[i].plot(t, noise_true, label=f"True noise")

    # cluttered env
    # for i, path in enumerate(cluttered_paths):
    #     t, safety_margin = safety_margin_helper(path)
    #     ax[1].plot(t, safety_margin, label=f"Robot {i}")

    # set other parameters
    for i in range(3):
        ax[i].legend()
        ax[i].grid()
        ax[i].set_xlabel("Time [s]")
        ax[i].set_ylabel("Noise [m]")

    plt.show()


def plot_noise_E1(gap_paths, cluttered_paths):
    # Same typography as safety-margin plot
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )

    W = 7.16  # full \textwidth in inches (IEEE)
    H = 0.50 * W
    fig, ax = plt.subplots(2, 1, figsize=(W, H), sharex=True, constrained_layout=True)

    # Gap environment
    for i, path in enumerate(gap_paths):
        t, noise_true, noise_est = noise_helper(path)
        ax[0].plot(
            t,
            noise_est,
            lw=1.0,
            label=rf"R{i} $\widetilde{{\sigma}}^\star(\widehat \mathbf{{x}})$",
        )
        ax[0].plot(
            t,
            noise_true,
            lw=1.0,
            linestyle="--",
            label=rf"R{i} ${{\sigma}}(\mathbf{{x}})$",
        )

    # Cluttered environment
    for i, path in enumerate(cluttered_paths):
        t, noise_true, noise_est = noise_helper(path)
        ax[1].plot(
            t,
            noise_est,
            lw=1.0,
            label=rf"R{i} $\widetilde{{\sigma}}^\star(\widehat \mathbf{{x}})$",
        )
        ax[1].plot(
            t,
            noise_true,
            lw=1.0,
            linestyle="--",
            label=rf"R{i} ${{\sigma}}(\mathbf{{x}})$",
        )

    # Styling identical to safety-margin plot
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.set_axisbelow(True)
        a.set_ylabel(r"Noise $\sigma$ [m]")
        a.margins(x=0)
        a.legend(loc="upper right", framealpha=0.9, ncol=2)

    ax[1].set_xlabel(r"$t$ [s]")
    ax[0].set_title(r"Noise $\sigma$ over time for Gap Environment")
    ax[1].set_title(r"Noise $\sigma$ over time for Cluttered Environment")

    fig.savefig("E1_noise.pdf", bbox_inches="tight", transparent=True)

def plot_noise_E3(gap_paths):
    # Same typography as safety-margin plot
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
        }
    )

    W = 7.16  # full \textwidth in inches (IEEE)
    H = 0.27 * W
    fig, ax = plt.subplots(1, 1, figsize=(W, H), sharex=True, constrained_layout=True)
    noise_values = [0.1, 0.15, 0.2]

    # Gap environment
    for i, path in enumerate(gap_paths):
        t, noise_true, noise_est = noise_helper(path)
        ax.plot(
            t,
            noise_est,
            lw=1.0,
            label=rf"$\widetilde{{\sigma}}^\star(\widehat \mathbf{{x}})$ $\sigma_{{\max}}={noise_values[i]}$",
        )
        ax.plot(
            t,
            noise_true,
            lw=1.0,
            linestyle="--",
            label=rf"${{\sigma}}(\mathbf{{x}})$ $\sigma_{{\max}}={noise_values[i]}$",
        )

    # Styling identical to safety-margin plot
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"$\sigma$ [m]")
    ax.margins(x=0)
    ax.legend(loc="upper right", framealpha=0.9, ncol=3)

    ax.set_xlabel(r"$t$ [s]")
    ax.set_title(
        r"Noise $\sigma$ over time for Gap Environment for different $\sigma_{\max}$"
    )
    # ax[1].set_title(r"Noise $\sigma$ over time for Gap Environment with $\sigma_{\max}=0.15$ m")
    # ax[2].set_title(r"Noise $\sigma$ over time for Gap Environment with $\sigma_{\max}=0.2$ m")

    fig.savefig("E3_noise.pdf", bbox_inches="tight", transparent=True)


# E1:
# 2x safety margin over time (for each env 1), only take seed 7 (othrwise to much overlap)
# 2x noise over time (for each env 1), only take seed 7 (otherwise to much overlap)
# run it
E1_gap_paths = [
    "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_0_seed_7/simulation_data",
    "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_1_seed_7/simulation_data",
    "./runs/E1_overall_performance/gap_env/simulation_results/gap_experiment_3_seed_7/simulation_data",
]
E1_cluttered_paths = [
    "./runs/E1_overall_performance/cluttered_env/simulation_results/cluttered_experiment_0_seed_7/simulation_data",
    "./runs/E1_overall_performance/cluttered_env/simulation_results/cluttered_experiment_1_seed_7/simulation_data",
    "./runs/E1_overall_performance/cluttered_env/simulation_results/cluttered_experiment_3_seed_7/simulation_data",
]
# plot_safety_margin_E1(E1_gap_paths, E1_cluttered_paths)
# plot_noise_E1(E1_gap_paths, E1_cluttered_paths)

# E2:
# plot percentile over time

# E3:
# noise and true noise over time
E3_gap_paths = [
    "./runs/E3_noise/noise_01/simulation_results/gap_experiment_3_seed_7/simulation_data",
    "./runs/E3_noise/noise_015/simulation_results/gap_experiment_3_seed_7/simulation_data",
    "./runs/E3_noise/noise_02/simulation_results/gap_experiment_3_seed_7/simulation_data",
]
# plot_noise_E3(E3_gap_paths)
plot_safety_margin_E3(E3_gap_paths)
