"""Generate the paper's budget table, transfer counts and fixed-work figure from the archive."""
import argparse
from fusionscope.common import ROOT,read,records

BOUND = ROOT/"results/partition_boundary"
SCOPE = ROOT/"results/scoped_assignment"
TEX = ROOT/"paper/partition_numbers.tex"
POLICY_LABEL = dict(native_J="native joint",separate="independent",packed="packed",
                    resource="resource rule",grain="grain rule",scoped_off="fixed scoped")


def macros():
    lines = []
    def command(key,val):
        lines.append(r"\newcommand{"+chr(92)+key+"}{"+str(val)+"}")
    for src,prefix in ((BOUND,"Packed"),(SCOPE,"Scoped")):
        root = src/"holdout/analysis"
        v = read(root/"verdict.json")
        command(prefix+"Ratio",f"{v['ratio']:.3f}")
        command(prefix+"Best",POLICY_LABEL[v["strongest_simple"]])
        command(prefix+"Valid",v["exact_cells"])
        if src==SCOPE:
            for macro,field in (("Faster","faster"),("Slower","slower"),("Tie","unresolved"),
                ("Harm","default_harms"),("Recovered","native_regressions_helped"),("Regress","native_regressions"),
                ("Wins","native_wins"),("Lost","native_wins_harmed"),("Par","native_assignment_wins"),
                ("ParLost","native_assignment_wins_harmed")):
                command(prefix+macro,v[field])
        totals = records(root/"cost_ratios.csv")
        for n,word in ((1,"One"),(100,"Hundred"),(1000,"Thousand"),(10000,"TenThousand")):
            row, = [r for r in totals if (r["policy"],r["baseline"],r["n"])==("shape_guard","native_J",str(n))]
            command(prefix+word+"Native",f"{float(row['ratio']):.3f}")
            if src==BOUND:
                row, = [r for r in totals if (r["policy"],r["baseline"],r["n"])==("packed","native_J",str(n))]
                command("FixedPacked"+word,f"{float(row['ratio']):.3f}")
            else:
                reuse, = [r for r in records(BOUND/"deployment/budgets.csv") if (r["n"],r["panel"],r["scenario"])
                          ==(str(n),"scoped_assignment","main_available")]
                command("Reusable"+word,f"{float(reuse['ratio_to_native']):.3f}")
        if src==SCOPE:
            for kind,label in (("residual_square","Residual"),("dual_input","Dual"),
                               ("gated_product","Gated"),("center_residual","Center")):
                command(prefix+label,f"{v['per_structure'][kind]:.3f}")
    return "\n".join(lines)+"\n"


def figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    rows = records(BOUND/"discovery/analysis/pairs.csv")
    chosen = [r for r in rows if r["left"]=="J" and r["right"]=="off" and r["kind"]=="minimal"
              and int(r["rows"])*int(r["width"])==46080]
    fig,ax = plt.subplots(figsize=(6.4,2.85),layout="constrained")
    for res,label,col,mark,offset in (("one","1 CPU","#0072B2","o",-.15),
          ("four","4 CPUs","#D55E00","s",0),("default","12 CPUs (default)","#00836D","^",.15)):
        xs = sorted((r for r in chosen if r["resource"]==res),key=lambda r:int(r["rows"]))
        loc = np.arange(len(xs))+offset
        ys = [float(r["delta_us"]) for r in xs]
        ax.plot(loc,ys,color=col,alpha=.5,lw=.9)
        for i,(y,r) in enumerate(zip(ys,xs,strict=True)):
            ax.scatter(loc[i],y,s=32,marker=mark,edgecolor=col,
                       facecolor="white" if r["verdict"]=="unresolved" else col,
                       label=label if i==1 else None,zorder=3)
    ax.axhline(0,color="0.25",lw=.8)
    ax.set_xticks(range(len(xs)),[r["rows"] for r in xs])
    ax.set_xlabel("Output rows at 46,080 input elements")
    ax.set_ylabel("Native minus pass off (µs)")
    ax.legend(frameon=False,ncols=3,loc="upper left",fontsize=8)
    ax.set_ylim(-21,149)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.16)
    fig.savefig(BOUND/"shape_boundary.png",dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("figure","tables","verify"))
    action = ap.parse_args().action
    if action=="figure":
        figure()
    elif action=="tables":
        TEX.write_text(macros(),encoding="utf8",newline="\n")
    else:
        assert TEX.read_text(encoding="utf8")==macros()


if __name__=="__main__":
    main()
