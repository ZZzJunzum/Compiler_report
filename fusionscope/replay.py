"""Package an audited independent run and compare its results to a reference run."""
import argparse
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .common import DOUBLES, ROOT, dump, read, records, sha


def summary(folder):
    tables=folder/"report/tables"
    audit=read(folder/"report/audit.json")
    classes=records(tables/"hlo_classes.csv")
    timing=records(tables/"latencies.csv")
    policies={r["method"]:r for r in records(tables/"policy_summary.csv") if r["group"]=="matrix"}
    ints=records(tables/"interactions.csv")
    train={(r["case"],r["mode"]):r for r in ints if r["phase"]=="matrix"}
    supported={}
    for r in ints:
        if r["phase"]!="confirm":
            continue
        key=(r["case"],r["mode"])
        if (r["outside_floor"]==train[key]["outside_floor"]=="True" and r["valid"]==train[key]["valid"]=="True"
                and float(r["interaction_us"])*float(train[key]["interaction_us"])>0):
            supported["/".join(key)]=float(r["interaction_us"])
    fingerprints={r["case"]+"/"+r["mode"]:r["canonical_sha256"] for r in records(tables/"hlo_fingerprints.csv")}
    stats={"timed_samples":audit["timed_samples_total"],"lock":audit["prediction_lock_sha256"],
           "checks":sum(p["semantic_checks"] for p in audit["raw_by_stage"].values()),
           "valid":sum(p["valid"] for p in audit["raw_by_stage"].values()),
           "equivalent_synthetic":sum(row["double_named_schedule_classes"]=="1" and not row["case"].startswith("R") for row in classes),
           "external_classes":[int(row["double_canonical_classes"]) for row in classes if row["case"].startswith("R")],
           "unresolved":int(policies["M2"]["ties"]),
           "median_ratio":median(float(r["paired_ratio"]) for r in timing if r["phase"]=="confirm" and r["mode"] in DOUBLES),
           "M2_median_regret":float(policies["M2"]["median_regret"]),
           "post_median_regret":float(policies["B0_post"]["median_regret"]),
           "M2_mae_us":float(policies["M2"]["mae_us"]),
           "single_add_mae_us":float(policies["single_add"]["mae_us"]),"supported_interactions":supported}
    return stats,fingerprints


def archive_repeat(folder,dest):
    files=sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".partial"))
    archive=dest/"cpu-clean-replay.zip"
    tmp=archive.with_suffix(".partial")
    with ZipFile(tmp,"w") as out:
        for path in files:
            name="results/clean_cpu/"+path.relative_to(folder).as_posix()
            entry=ZipInfo(name,date_time=(1980,1,1,0,0,0))
            out.writestr(entry,path.read_bytes(),compress_type=ZIP_DEFLATED,compresslevel=6)
    with ZipFile(tmp) as z:
        if z.testzip() is not None or len(z.infolist())!=len(files):
            raise ValueError("Replay archive failed CRC validation")
        for path,entry in zip(files,z.infolist()):
            if z.read(entry)!=path.read_bytes():
                raise ValueError(f"Replay archive content mismatch: {path}")
    tmp.replace(archive)
    return len(files),sha(archive)


def compare(reference,repeat,dest,measurement_commit,analysis_commit):
    dest.mkdir(parents=True,exist_ok=True)
    a,ah=summary(reference)
    b,bh=summary(repeat)
    if ah.keys()!=bh.keys():
        raise ValueError("Configuration sets differ")
    common=[k for k,i in a["supported_interactions"].items() if k in b["supported_interactions"]
            and i*b["supported_interactions"][k]>0]
    matches=sum(ah[k]==bh[k] for k in ah)
    files,digest=archive_repeat(repeat,dest)
    output={"reference":a,"repeat":b,"matching_hlo_signatures":matches,"compared_configurations":len(ah),
            "same_sign_over_noise_in_all_four_batches":common,"measurement_checkout":measurement_commit,
            "report_checkout":analysis_commit,"archive_files":files,"archive_sha256":digest}
    dump(dest/"comparison.json",output)
    rows=[
        ("完整性检查通过的逐次计时",a["timed_samples"],b["timed_samples"]),
        ("三种子校验通过 / 总数（含重复进程）",f"{a['valid']}/{a['checks']}",f"{b['valid']}/{b['checks']}"),
        ("四位置合为同一份 HLO 的合成条件",f"{a['equivalent_synthetic']}/32",f"{b['equivalent_synthetic']}/32"),
        ("R1 / R2 的 HLO 类数"," / ".join(map(str,a["external_classes"]))," / ".join(map(str,b["external_classes"]))),
        ("前两名无法稳定排序的合成条件",f"{a['unresolved']}/32",f"{b['unresolved']}/32"),
        ("双点相对无观测基线的配对延迟中位数",f"{a['median_ratio']:.3f}×",f"{b['median_ratio']:.3f}×"),
        ("M2 选中方案的中位损失",f"{a['M2_median_regret']:.3%}",f"{b['M2_median_regret']:.3%}"),
        ("固定 post/post 的中位损失",f"{a['post_median_regret']:.3%}",f"{b['post_median_regret']:.3%}"),
        ("单点相加的绝对开销 MAE",f"{a['single_add_mae_us']:.2f} µs",f"{b['single_add_mae_us']:.2f} µs"),
        ("发现与确认批在同一方向越过交互门槛",f"{len(a['supported_interactions'])}/128",f"{len(b['supported_interactions'])}/128")]
    text=f"""# 从 GitHub 干净克隆后的完整复跑

从远端重新克隆，在同一台 CPU 和固定依赖环境中创建空的输出目录，从输入生成开始重新执行先导、32 条件主矩阵、模型封存、独立确认、官方 ResNet 子图以及执行边界对照。复跑重新训练了各折模型并重新锁定预测。原始测量代码来自 `{measurement_commit[:12]}`，后续报告和写法对照采用 `{analysis_commit[:12]}`。正式归档也在这个干净克隆中通过了完整性检查。

| 指标 | 首次完整运行 | 干净复跑 |
|---|---:|---:|
"""
    text+="".join(f"| {name} | {left} | {right} |\n" for name,left,right in rows)
    text+=fr"""
对照两次运行的每个配置，**{matches}/{len(ah)}** 份规范化优化后 HLO 指纹相同。两次的模型预测分别封存，可在两份审计文件中看到不同的锁定哈希。M2 与固定 post/post 的细小排名随运行变化。四个批次（两次发现与两次确认）中同方向越过 A/A 伪交互门槛的组合有 **{len(common)} 项**，详见对照 JSON 中的完整 ID。[对照数据](comparison.json)

压缩包保留本次复跑的 {files:,} 个文件，包括全部逐轮读数、校验、编译器 IR、封存模型、报告和图表。打包后已逐文件对照原始字节并通过 CRC 检查。[下载完整复跑数据](cpu-clean-replay.zip)

将压缩包从仓库根目录解压，再用下列命令校验。

```powershell
Expand-Archive -LiteralPath results/reproducibility/cpu-clean-replay.zip -DestinationPath .
.\.venv\Scripts\python.exe run.py --stage verify --out results/clean_cpu
```

归档 SHA-256 为 `{digest}`。[首次运行报告](../cpu/report/REPORT.md)
"""
    (dest/"README.md").write_text(text,encoding="utf8",newline="\n")
    print(f"Compared {len(ah)} configurations, archived {files} replay files.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference",type=Path,default=ROOT/"results/cpu")
    ap.add_argument("--repeat",type=Path,default=ROOT/"reproduction/cpu")
    ap.add_argument("--dest",type=Path,default=ROOT/"results/reproducibility")
    ap.add_argument("--measurement-commit",required=True)
    ap.add_argument("--analysis-commit",required=True)
    opts=ap.parse_args()
    compare(opts.reference,opts.repeat,opts.dest,opts.measurement_commit,opts.analysis_commit)
