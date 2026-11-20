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
    out = ["The one-CPU panels have now run on "+str(len(eligible))+
           " distinct Chicago calendar dates. For each later axis batch we give joint minus separate "
           "for the global and row requests. For intervention batches we give the paired row-minus-global "
           "change on disabling assignment, followed by its repeatability floor. All values are in microseconds."]
    for date in eligible[1:]:
        for r in rows:
            if (r["local_date"],r["cpus"])!=(date,"1"):
                continue
            if (r["panel"],r["graph"])==("axis","t_E1_chain"):
                out.append(f"On {date} the full-chain global and row differences were "
                           f"{float(r['global_diff_us']):+.2f} and {float(r['row_diff_us']):+.2f}, "
                           f"with floors {float(r['global_diff_floor_us']):.2f} and {float(r['row_diff_floor_us']):.2f}.")
            elif r["panel"]=="task":
                name = {"t_E1_chain":"full chain","two_sites_large":"large reduced program",
                        "two_sites_small":"small reduced program"}[r["graph"]]
                out.append(f"The {name} interaction was {float(r['interaction_us']):+.2f} "
                           f"(floor {float(r['interaction_floor_us']):.2f}).")
    out.append("The calendar replication measures the original axis and intervention endpoints. "
               "The held-out policy panels still have one sampling date.")
    return " ".join(out)+"\n"


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify",action="store_true")
    if ap.parse_args().verify:
        assert TARGET.read_text(encoding="utf8")==render()
    else:
        TARGET.write_text(render(),encoding="utf8",newline="\n")
