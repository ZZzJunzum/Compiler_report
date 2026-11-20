"""Plot the November 20 discovery sessions from their archived contrasts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fusionscope.common import ROOT, records


def main():
    folder = ROOT/"results/axis_probe"
    rows = records(folder/"analysis/contrasts.csv") + records(folder/"jax082/analysis/contrasts.csv")
    lookup = {(r["session"], r["graph"]):r for r in rows}
    primary = "t_E1_chain"
    teal, orange = "#006968", "#ba501e"
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.35), gridspec_kw={"width_ratios":[1,1.17]})
    fig.subplots_adjust(left=.14, right=.985, top=.78, bottom=.23, wspace=.72)
    for ax in (left, right):
        ax.axvline(0, color="#393c40", lw=.8)
        ax.grid(axis="x", color="#e5e6e8", lw=.6)
        ax.set_axisbelow(True)
        ax.spines[["top","left","right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
    left.set_title("Joint minus separate", fontsize=11, loc="left", weight="bold")
    for y, session in zip((2,1,0), ("axis_1120_a", "axis_1120_b", "axis_1120_082")):
        r = lookup[session,primary]
        for offset, key, marker, color in ((.13,"global_diff","o",teal),(-.13,"row_diff","s",orange)):
            left.scatter(float(r[key+"_us"]), y+offset, marker=marker, color=color, s=42, zorder=3)
    left.set(yticks=[2,1,0], yticklabels=["0.9.0.1  A","0.9.0.1  B","0.8.2"], ylim=(-.46,2.55),
             xlim=(-67,85), xlabel="Paired latency difference (µs)")
    left.legend(handles=[Line2D([],[],ls="",marker="o",color=teal,label="Global"),
                         Line2D([],[],ls="",marker="s",color=orange,label="Row")],
                ncols=2,loc="lower left",bbox_to_anchor=(-.03,1.17),frameon=False,fontsize=8)
    right.set_title("Effect of disabling task assignment", fontsize=11, loc="left", weight="bold")
    for y, graph in zip((2,1,0),(primary,"two_sites_large","two_sites_small")):
        for offset, session, marker, color in ((.13,"task_1120_one","o",teal),(-.13,"task_1120_four","s",orange)):
            r = lookup[session,graph]
            floor = float(r["interaction_floor_us"])
            right.hlines(y+offset, -floor, floor, color="#9298a0", lw=3.0, zorder=2)
            right.scatter(float(r["interaction_us"]), y+offset, marker=marker, s=42, color=color, zorder=3)
    right.set(yticks=[2,1,0], yticklabels=["Full affine chain","Two sites, 288 rows","Two sites, 4 rows"],
              ylim=(-.46,2.55), xlim=(-133,16), xlabel="Change for row minus change for global (µs)")
    right.legend(handles=[Line2D([],[],ls="",marker="o",color=teal,label="1 logical CPU"),
                          Line2D([],[],ls="",marker="s",color=orange,label="4 allowed")],
                 ncols=2,loc="lower left",bbox_to_anchor=(-.03,1.17),frameon=False,fontsize=8)
    fig.savefig(folder/"first_day.png", dpi=220, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


if __name__ == "__main__":
    main()
