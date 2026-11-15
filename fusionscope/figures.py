"""Scientific figures drawn from archived HLO and paired measurements."""
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .common import DOUBLES, FAMILIES, conditions
from .hlo import Module, ancestors
from .stats import interactions

COLORS=["#176c85","#c95532","#6b579e","#3d8054"]
FAMILY_COLOR=dict(zip(FAMILIES,COLORS))


def save(fig,path):
    fig.savefig(path,dpi=155,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def mechanism(path,ir):
    g=Module(ir.read_text(encoding="utf8"))
    fig,ax=plt.subplots(figsize=(10,3.9))
    ax.set_xlim(-.55,4.54)
    ax.set_ylim(-.62,2.6)
    ax.axis("off")
    def box(x,y,title,line,color):
        ax.add_patch(FancyBboxPatch((x-.42,y-.22),.84,.44,boxstyle="round,pad=0.02,rounding_size=0.04",
                     edgecolor=color,facecolor="white",linewidth=1.4))
        ax.text(x,y+.07,title,ha="center",va="center",fontsize=10,color="#202832")
        ax.text(x,y-.09,f"HLO line {line}",ha="center",va="center",fontsize=8,color=color)
    for i,(name,out) in enumerate(zip(("y","q1","q2"),g.outputs)):
        route=ancestors(out)
        if i==0:
            nodes=[g.producer(g.relu("u")),g.relu("u"),g.producer(g.relu("v")),g.relu("v"),out]
            names=["u affine","ReLU 1","v affine","ReLU 2",name]
        else:
            cmp=next(v for v in route if v.op.kind=="compare")
            reds=sorted([v for v in route if v.op.kind in ("reduce-window","reduce")],key=lambda n:n.op.line)
            nodes=[cmp.parents[0],cmp,*reds]
            names=[("u" if i==1 else "v")+" affine","compare > 0",*[n.op.kind for n in reds]]
        y=2-i
        for j,(node,label) in enumerate(zip(nodes,names)):
            if j:
                ax.annotate("",(j-.46,y),(j-1+.44,y),arrowprops=dict(arrowstyle="->",color="#657582"))
            box(j,y,label,node.op.line,COLORS[i])
        if i:
            ax.text(len(nodes)-.49,y,name,fontsize=11,weight="bold",color=COLORS[i])
    ax.set_title("One main result and two count paths in optimized CPU HLO",loc="left",weight="bold",pad=8)
    fig.text(.11,.045,"E0 chain, post/post. Numbers refer to selected lines of the archived scheduled HLO.",fontsize=9)
    save(fig,path)


def figures(root,data,summaries):
    dest=root/"report/figures"
    dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
                        "axes.labelcolor":"#283747","text.color":"#263746","font.size":9})
    mechanism(dest/"01-hlo-paths.png",root/"matrix/ir/E0_chain/post_post/optimized_hlo.txt")
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout="constrained")
    for ax,f in zip(axes.flat,FAMILIES):
        cases=[c for c in conditions() if c[0]==f]
        for i,c in enumerate(cases):
            s=data["confirm",c]
            ys=[s["times"][m] for m in DOUBLES]
            lo=min(ys)
            ax.plot([0,s["aa_us"]*100/lo],[i,i],color="#c6cdd2",lw=6,solid_capstyle="butt")
            for j,(mode,y) in enumerate(zip(DOUBLES,ys)):
                ax.scatter(100*(y/lo-1),i+(j-1.5)*.105,s=16,color=COLORS[j],zorder=3,
                           label=mode.replace("_","/") if i==0 else "")
        ax.set_yticks(range(len(cases)),[c[1:].replace("_"," · ") for c in cases])
        ax.invert_yaxis()
        ax.set_title({"E":"Elementwise","M":"MLP","N":"Row normalization","C":"Residual convolution"}[f],
                     loc="left",fontweight="bold")
        ax.set_xlabel("Latency above fastest measured placement (%)")
        ax.grid(axis="x",alpha=.15)
    axes[0,0].legend(fontsize=7,ncol=2,loc="best")
    fig.suptitle("Equivalent placements in independent confirmation\nGrey bars show the same-executable A/A noise floor",fontsize=13,fontweight="bold")
    save(fig,dest/"02-placement-and-noise.png")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    all_t=[]
    for f in FAMILIES:
        xx,yy,rat=[],[],[]
        for c in (c for c in conditions() if c[0]==f):
            s=data["confirm",c]
            for row in interactions(s):
                m=row["mode"]
                u,v=m.split("_")
                xx.append(median(p["u_"+u]+p["v_"+v]-p["base"] for p in s["processes"]))
                yy.append(s["times"][m])
                if row["aa_pseudo_p95_us"]>0:
                    rat.append(row["interaction_us"]/row["aa_pseudo_p95_us"])
        all_t+=xx+yy
        ax.scatter(xx,yy,s=19,alpha=.7,color=FAMILY_COLOR[f],label=f)
        i=FAMILIES.index(f)
        offsets=np.linspace(-.19,.19,len(rat))
        bx.scatter(i+offsets,rat,alpha=.75,s=13,color=FAMILY_COLOR[f])
        bx.hlines(median(rat),i-.25,i+.25,color="#16232d",lw=2)
    ax.plot([min(all_t),max(all_t)],[min(all_t),max(all_t)],ls="--",color="#66747c",lw=1)
    ax.set(xscale="log",yscale="log",xlabel="Estimated from single sites (µs)",ylabel="Measured double-site latency (µs)")
    ax.legend(title="Family")
    ax.set_title("Measured latency vs. additive estimate",loc="left",fontweight="bold")
    bx.axhspan(-1,1,color="#dbe5e6",zorder=-1)
    bx.set_yscale("symlog",linthresh=1)
    bx.set_xticks(range(4),FAMILIES)
    bx.set_ylabel("Interaction / A/A pseudo-interaction threshold")
    bx.set_title("Departure from additivity",loc="left",fontweight="bold")
    save(fig,dest/"03-interactions.png")
    fig,ax=plt.subplots(figsize=(9,4.4),layout="constrained")
    methods=["B0_post","B2","M1","M2","single_add"]
    palette=["#596672",*COLORS]
    for i,method in enumerate(methods):
        values=[next(s for s in summaries if s["group"]==f and s["method"]==method) for f in FAMILIES]
        x=np.arange(4)+(i-2)*.15
        y=[v["median_regret"]*100 for v in values]
        tail=[v["p90_regret"]*100 for v in values]
        ax.vlines(x,y,tail,color=palette[i],alpha=.65,lw=1.5)
        ax.scatter(x,tail,marker="_",s=32,color=palette[i])
        ax.scatter(x,y,s=30,color=palette[i],label=method)
    ax.set_xticks(range(4),["Elementwise","MLP","Normalization","Convolution"])
    ax.set_ylabel("Gap from fastest valid placement (%)")
    ax.set_title("Choice on a held-out graph family",loc="left",fontsize=14,fontweight="bold")
    ax.legend(ncol=5,loc="upper center",bbox_to_anchor=(.5,-.1),fontsize=8)
    ax.text(.99,.99,"Dots: median    Upper ticks: 90th percentile",ha="right",va="top",transform=ax.transAxes,fontsize=8)
    ax.grid(axis="y",alpha=.15)
    save(fig,dest/"04-held-out-regret.png")
