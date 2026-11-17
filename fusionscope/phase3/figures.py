"""Publication figures generated directly from paired CPU evidence."""
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusionscope.common import records
from .design import MECHANISM, heldout


def render(out, decisions, budgets):
    dest=out/"figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans","pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    def save(fig,name):
        fig.savefig(dest/(name+".pdf"),bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None})
        plt.close(fig)
    rows=records(out/"analysis/mechanism.csv")
    fig,axes=plt.subplots(1,4,figsize=(9.5,2.55),layout="constrained")
    colors=["#50575b","#b05126","#126d74"]
    for ax,(c,sizes) in zip(axes,MECHANISM.items()):
        times={r["variant"]:float(r["us"]) for r in rows if r["case"]==c}
        positions=[*sizes,"Whole"]
        for i,(s,v,label) in enumerate(zip("BSJ",("V0","V1","V2"),("Main only","Separate","Joint"))):
            y=[times[s+str(b)] for b in sizes]+[times[v]]
            ax.plot(range(3),np.array(y)/times["V0"],marker=("o","s","^")[i],lw=1.4,color=colors[i],label=label)
        ax.axhline(1,color="#888888",lw=.5,ls=":")
        ax.set_xticks(range(3),positions)
        ax.set_title(c.replace("_"," "),fontsize=10)
        ax.set_xlabel("Tile extent")
        ax.set_yscale("log",base=2)
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}"))
    axes[0].set_ylabel("Time / whole main only")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,frameon=False)
    save(fig,"mechanism")
    fig,axes=plt.subplots(1,4,figsize=(9.6,2.9),sharey=True,layout="constrained")
    for ax,family in zip(axes,"EMNC"):
        cases=[c for c in heldout() if c.group=="primary" and c.ops[0]==family]
        for k,(policy,label,color) in enumerate((("V2","Fixed joint","#50575b"),("static","Locked static","#126d74"),
                                                  ("timed_rank_alt","Timed shortlist","#b05126"))):
            seq=[next(r for r in decisions if r["case"]==c.name and r["policy"]==policy) for c in cases]
            for j,r in enumerate(seq):
                if r["valid"]:
                    ax.scatter(j+(k-1)*.19,100*r["regret"],s=29,marker=("s","o","^")[k],
                               facecolor=color if not r["within_floor"] else "white",edgecolor=color,
                               label=label if j==0 else None,zorder=3)
        ax.set_title({"E":"Affine","M":"Dense","N":"Normalization","C":"Convolution"}[family])
        ax.set_xticks(range(6),["1c","1b","2c","2b","3c","3b"])
        ax.set_xlabel("Size and topology")
        ax.grid(axis="y",alpha=.22)
    axes[0].set_ylabel("Regret over valid best (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,frameon=False)
    save(fig,"regret")
    fig,axes=plt.subplots(1,2,figsize=(7.7,2.7),sharey=True,layout="constrained")
    for mode,ax in zip(("reused","cold"),axes):
        for policy,label,color,marker in (("static","Locked static","#126d74","o"),("timed_rank_alt","Timed shortlist","#b05126","^"),
                                         ("timed_fixed_alt","Timed fixed tile","#66549c","s")):
            seq=[r for r in budgets if r["group"]=="primary" and r["policy"]==policy and r["valid"]]
            joint={r["case"]:r for r in budgets if r["policy"]=="V2"}
            vals=[median(r[f"{mode}_H{h}_us"]/joint[r["case"]][f"{mode}_H{h}_us"] for r in seq) for h in (1,10,100,1000)]
            ax.plot((1,10,100,1000),vals,marker=marker,lw=1.4,color=color,label=label)
        ax.axhline(1,color="#50575b",ls=":",lw=1)
        ax.set(xscale="log",yscale="log",xlabel="Invocations",title="Calibration "+mode)
        ax.set_xticks((1,10,100,1000),("1","10","100","1000"))
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Median total time / fixed joint")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside upper center",ncol=3,frameon=False)
    save(fig,"reuse")
