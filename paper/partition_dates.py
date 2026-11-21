"""Render the manuscript's calendar-date replication panel from verified session results."""
import argparse
from fusionscope.common import ROOT,records

TARGET = ROOT/"paper/partition_dates.tex"


def render():
    rows = records(ROOT/"results/axis_probe/analysis/contrasts.csv")
    dates = sorted({r["local_date"] for r in rows if r["panel"]=="axis" and r["cpus"]=="1"})
    tasks = {r["local_date"] for r in rows if r["panel"]=="task" and r["cpus"]=="1"}
    eligible = [date for date in dates if date in tasks]
    if len(eligible)<2:
        return ("The archive currently contains the original one-CPU axis and large/small intervention "
                "panels on November 20, 2026. The predeclared second calendar date has no observations "
                "in this version. The independent structures below were timed on the first date.\n")
    first = [r for r in rows if (r["local_date"],r["cpus"],r["panel"],r["graph"])
             ==(eligible[0],"1","axis","t_E1_chain")]
    first_values = " and ".join(f"${float(r['interaction_us']):+.2f}$" for r in first)
    out = [f"On {eligible[0]} the full-chain row-minus-global interactions for joint versus separate "
           f"counting were {first_values} microseconds in the original one-CPU sessions."]
    for date in eligible[1:]:
        for r in rows:
            if (r["local_date"],r["cpus"])!=(date,"1"):
                continue
            if (r["panel"],r["graph"])==("axis","t_E1_chain"):
                out.append(f"On {date}, joint minus separate was ${float(r['global_diff_us']):+.2f}$ for global counts "
                           f"and ${float(r['row_diff_us']):+.2f}$ for row counts. The paired interaction reached "
                           f"${float(r['interaction_us']):+.2f}$ (floor {float(r['interaction_floor_us']):.2f}). "
                           "In the pass intervention we measure the row-minus-global change on disabling assignment.")
            elif r["panel"]=="task":
                name = {"t_E1_chain":"full chain","two_sites_large":"large reduced program",
                        "two_sites_small":"small reduced program"}[r["graph"]]
                out.append(f"The {name} interaction was ${float(r['interaction_us']):+.2f}$ "
                           f"(floor {float(r['interaction_floor_us']):.2f}).")
    out.append("These differences and floors are in microseconds. "
               "The held-out policy panels were measured on the first date.")
    return " ".join(out)+"\n"


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify",action="store_true")
    if ap.parse_args().verify:
        assert TARGET.read_text(encoding="utf8")==render()
    else:
        TARGET.write_text(render(),encoding="utf8",newline="\n")
