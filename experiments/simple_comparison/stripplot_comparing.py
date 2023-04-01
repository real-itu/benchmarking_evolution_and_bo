"""
Makes a stripplot comparing the number of point evaluations
of the objective function for both CMA-ES and BO.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")
sns.set(font_scale=1.5)

FIG_PATH = Path("/Users/migd/Projects/dissertation/Figures/Chapter_3/bo_related_plots")


if __name__ == "__main__":
    path_ = Path(__file__).parent.resolve()

    # reading the cma-es experiments
    easom_cmaes = pd.read_csv(path_ / "easom_cmaes.csv", index_col=0)
    cross_cmaes = pd.read_csv(path_ / "cross_cmaes.csv", index_col=0)

    easom_cmaes["Obj. function"] = ["easom"] * len(easom_cmaes)
    easom_cmaes["Algorithm"] = ["CMA-ES"] * len(easom_cmaes)

    print(f"EASOM CMA-ES: {len(easom_cmaes)}, mean: {easom_cmaes['num_points'].mean()}")

    cross_cmaes["Obj. function"] = ["cross-in-tray"] * len(cross_cmaes)
    cross_cmaes["Algorithm"] = ["CMA-ES"] * len(cross_cmaes)

    print(f"CROSS CMA-ES: {len(cross_cmaes)}, mean: {cross_cmaes['num_points'].mean()}")
    # reading the bo experiments
    easom_bo = pd.read_csv(path_ / "easom_bo.csv", index_col=0)
    cross_bo = pd.read_csv(path_ / "cross_bo.csv", index_col=0)

    easom_bo["Obj. function"] = ["easom"] * len(easom_bo)
    easom_bo["Algorithm"] = ["BO"] * len(easom_bo)
    print(f"EASOM BO: {len(easom_bo)}, mean: {easom_bo['num_points'].mean()}")

    cross_bo["Obj. function"] = ["cross-in-tray"] * len(cross_bo)
    cross_bo["Algorithm"] = ["BO"] * len(cross_bo)
    print(f"CROSS BO: {len(cross_bo)}, mean: {cross_bo['num_points'].mean()}")

    # Joining all these together
    df = pd.concat((easom_cmaes, easom_bo, cross_cmaes, cross_bo), axis=0)
    df.rename({"n_points": "Num. points"}, axis=1, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.stripplot(data=df, x="Obj. function", y="num_points", hue="Algorithm", size=8)
    ax.set_ylabel("Num. points queried")
    ax.legend(loc="upper right")
    fig.savefig(FIG_PATH / "comparing_bo_and_cmaes.jpg", dpi=120, bbox_inches="tight")
    plt.show()
