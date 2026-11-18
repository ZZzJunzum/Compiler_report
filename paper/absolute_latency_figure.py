"""Render the frozen pre-compilation estimator against measured whole-forward latency."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusionscope.common import ROOT, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"results/cpu_cost")
    args = parser.parse_args()
    rows = [r for r in records(args.data/"analysis/measurements.csv") if r["variant"]!="V0"]
    fig, ax = plt.subplots(figsize=(3.45,2.45),layout="constrained")
    plt.rcParams["pdf.fonttype"] = 42
    for group,label,mark,color in [("new_shape","New sizes","o","#1c5880"),
                                 ("new_architecture","New architectures","x","#b44024")]:
        subset = [r for r in rows if r["group"]==group]
        ax.scatter([float(r["runtime_us"]) for r in subset],
                   [float(r["combined_runtime_us"]) for r in subset],
                   marker=mark,c=color,s=10,alpha=.65,linewidths=.7,label=label,rasterized=False)
    ends = [float(r[k]) for r in rows for k in ("runtime_us","combined_runtime_us")]
    lo, hi = min(ends)*.75, max(ends)*1.22
    ax.fill_between([lo,hi],[lo/2,hi/2],[2*lo,2*hi],color=".92",zorder=-2)
    ax.plot([lo,hi],[lo,hi],color=".25",lw=.8,ls="--",zorder=-1)
    ax.set(xscale="log",yscale="log",xlim=(lo,hi),ylim=(lo,hi),
           xlabel="Measured latency (µs)",ylabel="Predicted latency (µs)")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.legend(loc="upper left",fontsize=7,frameon=False,handlelength=1.1,borderaxespad=.35)
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(args.data/"absolute_latency.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__ == "__main__":
    main()
