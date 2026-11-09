"""从现有 CSV 与场景生成代码绘制报告图；不重新运行算法或修改实验数据。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from extension_experiments import alternating_terrain, narrow_corridor
from innovation_experiments import striped_grid


ROOT = Path(__file__).parent
DEFAULT_OUTPUT = ROOT / "results" / "final_figures"
BLUE = "#2563A6"       # 基线
ORANGE = "#E68632"     # 优化
TEAL = "#438F87"       # 控制组
RED = "#BE5D54"        # 不利顺序
GRAY = "#6B7280"
LIGHT = "#E5EAF0"


def style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.titlesize": 14,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#77808A",
        "axes.linewidth": 0.8,
    })


def tidy(ax, axis="y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=LIGHT, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"空 CSV：{path}")
    return rows


def one(rows, **filters):
    selected = [row for row in rows if all(row.get(key) == str(value) for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"期望唯一数据行：{filters}，实际 {len(selected)} 行")
    return selected[0]


def number(row, field) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"缺少 {field}：{row}")
    return float(value)


def change(base, improved, metric) -> float:
    old, new = number(base, metric), number(improved, metric)
    if metric == "mean_seconds":
        old, new = old * 1000, new * 1000  # 先统一换成毫秒，再计算时间变化率。
    if old == 0:
        raise ValueError(f"{metric} 的基线为零，无法计算变化率")
    return (new - old) / old * 100


def checked_pair(rows, base_name, improved_name, *, case, depth="", metric="visited"):
    base = one(rows, case=case, algorithm=base_name, depth=depth)
    improved = one(rows, case=case, algorithm=improved_name, depth=depth)
    for field in ("task", "case", "depth", "seed", "repeats", "result", "best_target"):
        if base.get(field, "") != improved.get(field, ""):
            raise ValueError(f"不可比：{case} {base_name}/{improved_name} 的 {field} 不同")
    number(base, metric)
    number(improved, metric)
    return base, improved


def bars(ax, positions, values, width, label, color, *, offset=0, fmt="%.0f") -> None:
    artists = ax.bar(np.asarray(positions) + offset, values, width, label=label, color=color,
                     edgecolor="white", linewidth=0.4)
    ax.bar_label(artists, labels=[fmt % value for value in values], padding=2, fontsize=8)


def save(fig, output: Path, filename: str) -> Path:
    fig.savefig(output / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output / filename


def figure1(stage, output):
    cases = [("pdf_maze", "PDF 5×5"), ("random_maze_20x20", "随机 20×20\nseed=2026")]
    pairs = [(one(stage, case=case, task=1, parameter="zero"),
              one(stage, case=case, task=1, parameter="manhattan_lower_bound")) for case, _ in cases]
    for a, b in pairs:
        if a["result"] != b["result"] or a["repeats"] != b["repeats"]:
            raise ValueError("题一启发式对照的结果或重复次数不一致")
    x = np.arange(len(cases))
    base = [number(a, "expanded") for a, _ in pairs]
    guided = [number(b, "expanded") for _, b in pairs]
    rates = [-change(a, b, "expanded") for a, b in pairs]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    bars(top, x, base, 0.36, "h=0（Dijkstra）", BLUE, offset=-0.18)
    bars(top, x, guided, 0.36, "曼哈顿启发式", ORANGE, offset=0.18)
    top.set(ylabel="扩展节点数", title="不同迷宫场景的搜索开销")
    top.set_ylim(0, max(base) * 1.2)
    top.legend()
    tidy(top)
    bars(bottom, x, rates, 0.55, "节点减少率", ORANGE, fmt="%.1f%%")
    bottom.set(ylabel="节点减少率（%）", xlabel="实验场景", xticks=x,
               xticklabels=[label for _, label in cases], title="相对 h=0 的扩展节点减少率")
    bottom.set_ylim(0, max(rates) * 1.4)
    tidy(bottom)
    fig.suptitle("图1  A* 启发式搜索效率：PDF 迷宫与固定随机迷宫（第二阶段，各20次）")
    return save(fig, output, "图1_A星启发式效率对比.png")


def figure2(stage, output):
    cases = [("pdf_tree", "PDF 博弈树"), ("full_tree_d6_b4", "深度6／分支4\nseed=2026")]
    algorithms = [("Minimax", "left_to_right", "Minimax", BLUE),
                  ("Alpha-Beta", "left_to_right", "Alpha-Beta 默认", ORANGE),
                  ("Alpha-Beta", "first_leaf", "Alpha-Beta 首叶排序", TEAL)]
    vals = []
    rates = []
    for case, _ in cases:
        records = [one(stage, case=case, algorithm=algo, order=order) for algo, order, _, _ in algorithms]
        if len({r["result"] for r in records}) != 1 or len({r["repeats"] for r in records}) != 1:
            raise ValueError(f"题三结果或重复次数不一致：{case}")
        vals.append([number(r, "visited") for r in records])
        rates.append([0] + [-change(records[0], r, "visited") for r in records[1:]])
    x = np.arange(2)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for i, (_, _, label, color) in enumerate(algorithms):
        bars(axes[0], x, [vals[j][i] for j in range(2)], 0.25, label, color,
             offset=(i - 1) * 0.25)
    axes[0].set(ylabel="访问节点数", title="Minimax 与 Alpha-Beta 的实际访问量")
    axes[0].set_ylim(0, max(v[0] for v in vals) * 1.2)
    axes[0].legend(ncol=3, loc="upper center")
    tidy(axes[0])
    for i in (1, 2):
        _, _, label, color = algorithms[i]
        bars(axes[1], x, [rates[j][i] for j in range(2)], 0.30, label, color,
             offset=(-0.15 if i == 1 else 0.15), fmt="%.1f%%")
    axes[1].set(ylabel="相对 Minimax 减少率（%）", xlabel="博弈树场景", xticks=x,
                xticklabels=[label for _, label in cases], title="保持根值不变时的节点减少率")
    axes[1].set_ylim(0, max(v[1] for v in rates) * 1.35)
    tidy(axes[1])
    fig.suptitle("图2  Alpha-Beta 剪枝效率（第二阶段，每配置20次）")
    return save(fig, output, "图2_AlphaBeta剪枝效率对比.png")


def figure3(innovation, output):
    comparisons = [
        ("题一", "pdf_maze", "", "A*_FIFO", "A*_deeper_tie", "PDF迷宫", "expanded"),
        ("题一", "open_25x25", "", "A*_FIFO", "A*_deeper_tie", "空旷25×25", "expanded"),
        ("题一", "random_20x20", "", "A*_FIFO", "A*_deeper_tie", "随机20×20", "expanded"),
        ("题二", "pdf_weighted", "", "A*_global_min", "A*_axis_min", "PDF加权", "expanded"),
        ("题二", "striped_25x25", "", "A*_global_min", "A*_axis_min", "条纹25×25", "expanded"),
        ("题二", "random_30x30", "", "A*_global_min", "A*_axis_min", "随机30×30", "expanded"),
        ("题三", "pdf_tree", "", "Alpha-Beta", "Alpha-Beta_TT", "PDF树／置换表", "visited"),
        ("题三", "shared_dag", "", "Alpha-Beta", "Alpha-Beta_TT", "共享DAG／置换表", "visited"),
        ("题四", "pdf_treasure", "2", "Minimax_no_cache", "Minimax_memo", "PDF H2／缓存", "visited"),
        ("题四", "pdf_treasure", "4", "Minimax_no_cache", "Minimax_memo", "PDF H4／缓存", "visited"),
        ("题四", "six_treasure_grid", "4", "Minimax_no_cache", "Minimax_memo", "六宝藏 H4／缓存", "visited"),
        ("题四", "six_treasure_grid", "6", "Minimax_no_cache", "Minimax_memo", "六宝藏 H6／缓存", "visited"),
    ]
    colors = {"题一": BLUE, "题二": ORANGE, "题三": TEAL, "题四": "#8068A6"}
    marks = {"题一": "o", "题二": "s", "题三": "D", "题四": "^"}
    fig, ax = plt.subplots(figsize=(11.5, 7.2), layout="constrained")
    seen = set()
    for task, case, depth, old, new, label, metric in comparisons:
        a, b = checked_pair(innovation, old, new, case=case, depth=depth, metric=metric)
        if a["repeats"] != "50":
            raise ValueError("图3只使用 innovations 同批次50次计时")
        x, y = change(a, b, metric), change(a, b, "mean_seconds")
        ax.scatter(x, y, s=73, marker=marks[task], color=colors[task],
                   edgecolor="white", linewidth=0.8, zorder=4,
                   label=task if task not in seen else None)
        seen.add(task)
        ax.annotate(label, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set(xlabel="节点数变化率（%，负值表示优化减少）",
           ylabel="平均运行时间变化率（%，按 ms 计算）",
           title="同一场景内：优化方法相对各自基线的变化")
    ax.legend(title="任务", ncol=4, loc="upper left")
    tidy(ax, "both")
    fig.suptitle("图3  四项任务优化效果（仅 innovations 同批次、每配置50次）")
    return save(fig, output, "图3_四项任务优化效果综合.png")


def figure4(heuristic, output):
    records = [r for r in heuristic if r["case"] == "pdf_treasure"]
    depths = sorted({int(r["depth"]) for r in records})
    algorithms = [("current_score", "当前收益评价", BLUE, "o"),
                  ("future_estimate", "加入未来估值", ORANGE, "s")]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.1), sharex=True, layout="constrained")
    for algo, label, color, marker in algorithms:
        selected = [one(records, algorithm=algo, depth=d) for d in depths]
        if any(r["repeats"] != "50" for r in selected):
            raise ValueError("图4深度组重复次数不符")
        targets = [int(r["best_target"].removeprefix("T")) for r in selected]
        actual = [number(r, "realized_value") for r in selected]
        axes[0].plot(depths, targets, color=color, marker=marker, linewidth=1.8, markersize=7, label=label)
        axes[1].plot(depths, actual, color=color, marker=marker, linewidth=1.8, markersize=7, label=label)
    axes[0].set(ylabel="首步目标", yticks=[1, 2, 3, 4], yticklabels=["T1", "T2", "T3", "T4"],
                title="深度截止评价改变了哪些首步？")
    axes[0].legend(loc="upper left")
    axes[1].set(xlabel="搜索深度 H", ylabel="固定首步后的完整搜索收益差",
                title="首步决策质量：继续完整对抗后的实际效用", xticks=depths)
    axes[1].axhline(number(records[0], "full_optimal_value"), color=GRAY, linestyle="--",
                    linewidth=1, label="完整搜索最优值")
    axes[1].legend(loc="lower right")
    for ax in axes:
        tidy(ax)
    fig.suptitle("图4  PDF 四宝藏：H=1–4 的首步与实际收益（估值实验，每配置50次）")
    return save(fig, output, "图4_宝藏搜索深度与决策变化.png")


def figure5(extension, output):
    base = one(extension, case="Q3-01_natural", algorithm="Alpha-Beta", depth=6)
    variants = [("Q3-01_natural", "原顺序", BLUE, "o"),
                ("Q3-02_favorable", "事后有利顺序", ORANGE, "*"),
                ("Q3-03_unfavorable", "事后不利顺序", RED, "X")]
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    for case, label, color, marker in variants:
        row = one(extension, case=case, algorithm="Alpha-Beta", depth=6)
        if (row["seed"], row["repeats"], row["result"]) != (base["seed"], base["repeats"], base["result"]):
            raise ValueError("题三排序对照不是同一种子、重复次数或根值")
        x, y = change(base, row, "visited"), change(base, row, "mean_seconds")
        ax.scatter(x, y, s=160 if marker == "*" else 105, marker=marker, color=color,
                   edgecolor="white", linewidth=0.7, label=label, zorder=4)
        ax.annotate(label, (x, y), xytext=(7, 6), textcoords="offset points")
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set(xlabel="相对原顺序的访问节点变化率（%）",
           ylabel="相对原顺序的平均耗时变化率（%，按 ms 计算）",
           title="同一棵深度6、分支4博弈树；seed=2026\n有利／不利顺序按真实 Minimax 值事后构造，不含构造时间")
    tidy(ax, "both")
    ax.legend(loc="upper left")
    fig.suptitle("图5  Alpha-Beta 对节点访问顺序的敏感性（拓展实验，每配置20次）")
    return save(fig, output, "图5_AlphaBeta节点排序适用性.png")


def figure6(extension, innovation, output):
    corridor = narrow_corridor()
    open_grid = [["S" if (r, c) == (0, 0) else "E" if (r, c) == (24, 24) else "."
                  for c in range(25)] for r in range(25)]  # innovation_experiments.run 的原场景定义
    grids = [(corridor, "蛇形单通路", extension, "Q1-01_corridor_25"),
             (open_grid, "空旷多路径", innovation, "open_25x25")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.3), layout="constrained",
                             gridspec_kw={"height_ratios": [1.5, 1]})
    for col, (grid, title, rows, case) in enumerate(grids):
        image = np.array([[0 if cell == "X" else 1 for cell in row] for row in grid])
        axes[0, col].imshow(image, cmap=ListedColormap(["#39434E", "#F4F7FA"]),
                            vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].scatter([0, 24], [0, 24], c=[ORANGE, BLUE], s=55, marker="s")
        axes[0, col].annotate("S", (0, 0), xytext=(6, 5), textcoords="offset points", color=ORANGE,
                              fontsize=10, fontweight="bold")
        axes[0, col].annotate("E", (24, 24), xytext=(-16, -12), textcoords="offset points", color=BLUE,
                              fontsize=10, fontweight="bold")
        axes[0, col].set(title=f"{title}（25×25）", xlabel="列", ylabel="行", xticks=[0, 12, 24], yticks=[0, 12, 24])
        base, new = checked_pair(rows, "A*_FIFO", "A*_deeper_tie", case=case, metric="expanded")
        vals = [number(base, "expanded"), number(new, "expanded")]
        bars(axes[1, col], [0, 1], vals, 0.62, "扩展节点", [BLUE, ORANGE])
        axes[1, col].set(xticks=[0, 1], xticklabels=["FIFO", "大 g 优先"], ylabel="扩展节点数",
                          ylim=(0, max(vals) * 1.18), title=f"最短路径代价 {base['result']}")
        tidy(axes[1, col])
    fig.suptitle("图6  迷宫结构与 A* 平局策略：地图来自现有场景代码，节点来自 CSV")
    return save(fig, output, "图6_迷宫结构与搜索效率.png")


def figure7(extension, innovation, output):
    cases = [(striped_grid(), "高价条纹（25×25）", innovation, "striped_25x25", 71),
             (alternating_terrain(), "高低交错（25×25）", extension, "Q2-02_alternating_25", 239)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2), layout="constrained",
                             gridspec_kw={"height_ratios": [1.55, 1]})
    heat = None
    for col, (grid, label, rows, case, expected_cost) in enumerate(cases):
        weights = np.array([[0 if cell == "E" else 1 if cell == "S" else int(cell)
                             for cell in row] for row in grid])
        heat = axes[0, col].imshow(weights, cmap="YlOrBr", vmin=0, vmax=9, interpolation="nearest")
        axes[0, col].scatter([0, 24], [0, 24], c=[BLUE, RED], s=55, marker="s", edgecolors="white")
        axes[0, col].set(title=label, xlabel="列", ylabel="行", xticks=[0, 12, 24], yticks=[0, 12, 24])
        names = [("A*_global_min", "全图最低费", BLUE),
                 ("A*_global_min_deeper", "最低费＋同分深入", TEAL),
                 ("A*_axis_min", "行列下界", ORANGE)]
        if col == 1:
            names.insert(0, ("Dijkstra_h0", "h=0", GRAY))
        records = [one(rows, case=case, algorithm=name) for name, _, _ in names]
        if any(number(row, "result") != expected_cost for row in records):
            raise ValueError("题二地图的路径代价与记录不符")
        bars(axes[1, col], list(range(len(names))), [number(r, "expanded") for r in records],
             0.65, "扩展节点", [color for _, _, color in names])
        axes[1, col].set(xticks=list(range(len(names))), xticklabels=[name for _, name, _ in names],
                          ylabel="扩展节点数", ylim=(0, max(number(r, "expanded") for r in records) * 1.20),
                          title=f"最优路径代价 {expected_cost}")
        axes[1, col].tick_params(axis="x", labelrotation=12, labelsize=9)
        tidy(axes[1, col])
        if col == 0:
            axes[1, col].text(0.02, 0.98, "h=0：未测", transform=axes[1, col].transAxes,
                              ha="left", va="top", fontsize=9, color=GRAY)
    fig.colorbar(heat, ax=list(axes[0]), shrink=0.75, pad=0.02, label="进入格子的代价")
    fig.suptitle("图7  加权地形与启发函数：条纹图复用旧 CSV，交错图来自拓展实验；E 免费进入")
    return save(fig, output, "图7_加权地形与启发式效果.png")


def figure8(extension, output):
    algorithms = [("minimax_natural", "Minimax", BLUE),
                  ("alpha_beta_natural", "Alpha-Beta 默认", TEAL),
                  ("alpha_beta_gain", "Alpha-Beta 收益排序", ORANGE)]
    panels = [
        ("固定搜索深度 H=4", [("Q4-01_pdf_4", 4, "4个\nPDF地图"),
                              ("Q4-02_grid_6", 4, "6个\n8×8网格"),
                              ("Q4-03_grid_8", 4, "8个\n8×8网格")]),
        ("完整搜索深度 H=宝藏数", [("Q4-01_pdf_4", 4, "4个\nH=4"),
                                    ("Q4-02_grid_6", 6, "6个\nH=6"),
                                    ("Q4-03_grid_8", 8, "8个\nH=8")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), layout="constrained")
    for ax, (title, cases) in zip(axes, panels):
        x = np.arange(3)
        for index, (algorithm, label, color) in enumerate(algorithms):
            values = [number(one(extension, case=case, algorithm=algorithm, depth=depth), "visited")
                      for case, depth, _ in cases]
            bars(ax, x, values, 0.25, label, color, offset=(index - 1) * 0.25)
        for case, depth, _ in cases:
            rows = [one(extension, case=case, depth=depth, algorithm=name) for name, _, _ in algorithms]
            if len({r["result"] for r in rows}) != 1:
                raise ValueError(f"题四算法结果不一致：{case}, H={depth}")
        ax.set(yscale="log", ylabel="访问节点数（对数刻度）", xticks=x,
               xticklabels=[label for _, _, label in cases], title=title)
        ax.legend(loc="upper left")
        tidy(ax)
        ax.grid(which="minor", axis="y", color="#F0F2F4", linewidth=0.5)
    fig.suptitle("图8  宝藏数量、搜索深度与节点规模（拓展实验，每配置20次）")
    return save(fig, output, "图8_宝藏数量与搜索规模增长.png")


def figure10(matrix_rows, output):
    names = list(matrix_rows[0])[1:]
    if len(names) != 6 or [row["from"] for row in matrix_rows] != names:
        raise ValueError("距离矩阵不是 A、B、T1–T4 的 6×6 格式")
    data = np.array([[number(row, name) for name in names] for row in matrix_rows])
    fig, ax = plt.subplots(figsize=(7.2, 6.3), layout="constrained")
    heat = ax.imshow(data, cmap="Blues", vmin=0, vmax=float(data.max()))
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                    color="white" if data[i, j] > data.max() * 0.58 else "#1F2937", fontsize=10)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=names, yticklabels=names,
           xlabel="目标关键点", ylabel="出发关键点", title="A、B 与四个宝藏之间的最小移动代价")
    fig.colorbar(heat, ax=ax, label="最小移动代价")
    fig.suptitle("图10  PDF 宝藏地图的 6×6 距离矩阵（直接读取 CSV）")
    return save(fig, output, "图10_宝藏间距离矩阵热力图.png")


def description(output: Path) -> Path:
    content = """# 实验二图表说明

所有图片由 `python plot_results.py` 从现有 CSV 和原场景生成代码绘制，保存在本目录；**没有重跑搜索或修改实验结果**。PNG 为 240 dpi。柱高、散点位置、减少率和收益均在绘图时读取或计算。蓝色通常表示基线，橙色表示优化策略；其他颜色用于控制组或不利顺序。

## 数据口径

| 批次 | 文件 | 重复次数 | 本图中的用途 |
| --- | --- | ---: | --- |
| 第二阶段 | `../stage2/summary.csv` | 20 | 图1、图2的确定性节点数。 |
| 创新实验 | `../innovations/summary.csv` | 50 | 图3同批次的节点与平均时间；图6空旷图、图7条纹图的确定性节点数。 |
| 估值实验 | `../innovations/heuristic_evaluation_summary.csv` | 50 | 图4首步与固定首步后的完整搜索收益。 |
| 拓展实验 | `../extension/summary.csv` | 20 | 图5、图8同批次对照；图6单通路、图7交错图的确定性节点数。 |
| 场景定义 | `extension_experiments.py`、`innovation_experiments.py` | — | 图6地图、图7地形热力图。 |
| 距离矩阵 | `../stage2/distance_matrix.csv` | — | 图10矩阵单元格。 |

`../extension/coverage.csv` 与 `cases.csv` 核对了新测、复用和仅测试的场景。`stage2` 的时间来自算法内部计时，`innovations` 和 `extension` 记录完整函数调用；图中**不跨批次比较运行时间**。各图的节点数含义依任务区分：A* 使用 `expanded`，博弈树/宝藏搜索使用 `visited`，没有把 `pruned` 直接当作跳过的所有后代。时间先从秒统一换算为 ms，再计算变化率；变化率是 `(优化－基线)/基线×100%`，负值代表减少。减少率图则显示该数的相反数。按此公式，节点数和时间都下降的理想区域是**左下象限**；任务说明中的“右下象限”与给定公式方向不一致。

## 逐图说明

| 图 | 来源 | 主要发现与阅读限制 |
| --- | --- | --- |
| 图1 A* 启发式效率 | 第二阶段 `summary.csv`，题一 PDF 与随机20×20 | 曼哈顿启发式在这两图减少扩展数。场景结构不同，不是纯规模增长实验。 |
| 图2 Alpha-Beta 剪枝 | 第二阶段 `summary.csv`，PDF 树与深度6/分支4树 | Alpha-Beta 保持根值并减少访问；首叶排序在较大树中未必优于默认顺序。 |
| 图3 四任务优化效果 | 创新实验 `summary.csv`，同场景配对 | 每点代表一个基线/优化对，横轴节点变化、纵轴平均时间变化。题三置换表的 `visited` 不计缓存命中，反映实际求值量；不能当成全部函数调用次数。不同任务仅比较百分比，绝对节点数不可混用。 |
| 图4 搜索深度与决策 | 估值实验 `heuristic_evaluation_summary.csv`，PDF四宝藏 H=1–4 | 上图为所选首步，下图是该首步后完整对抗搜索的 `realized_value`；它不是深度截止的 `estimate_value`。未来估值在 H=3 改善决策，但不能保证普遍改善。 |
| 图5 节点顺序适用性 | 拓展实验 `summary.csv`，同一棵 seed=2026 的树 | 有利排序减少访问与时间，不利排序增加开销。两种排序用真实子树 Minimax 值**事后构造**，其构造时间不在图中，不是可免费使用的实际算法。 |
| 图6 迷宫结构 | `extension_experiments.narrow_corridor`、创新实验中的空旷地图构造；拓展及创新 `summary.csv` | 单通路两策略均扩展337；空旷图 FIFO 625、大 g 优先49。两批次仅比较确定性节点数，不比较时间；两地图最短路径长度也不同。 |
| 图7 地形分布 | `striped_grid`、`alternating_terrain`；创新及拓展 `summary.csv` | 条纹图行列下界扩展49，交错图为625。条纹图**没有 h=0 记录**，因此未绘该柱。热力图 S 按权值1着色，E 按免费终点代价0着色；路径代价分别为71、239。预处理耗时需看各批次内 CSV，不能横跨批次比较。 |
| 图8 宝藏规模 | 拓展实验 `summary.csv`，4/6/8宝藏 | 左边固定 H=4，右边搜完全部宝藏；对数纵轴。4宝藏采用 PDF 7×7，6/8宝藏采用无障碍 8×8，规模差异同时伴随地图差异。 |
| 图10 距离矩阵 | 第二阶段 `distance_matrix.csv` | 每格为从行关键点到列关键点的实算最小移动代价；与宝藏价值、最终收益不同。 |

## 未生成图9

图9是可选的宝藏决策树示意图。现有 `solve()` 仅记录访问总数、叶数、剪枝直接分支数，不保存每次节点访问、界限变化和被剪枝分支的完整轨迹。仅用汇总 CSV 绘制枝条会成为推测，因此本轮没有生成；若后续需要，应先在算法中增加可复现的 trace 导出。

## 建议用于五页内报告

优先选 **图2、图3、图5、图7、图8**：分别支撑剪枝效率、优化收益与代价、排序适用条件、启发函数局限和问题规模影响。需要解释深度受限决策时再选图4；图1、图6与图10可按篇幅作为补充。报告文字应同时写出图3中节点减少但时间增加的点，以及图5的事后排序限制。
"""
    path = output / "图表说明.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="从现有实验 CSV 生成最终论文风格图表")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    style()
    stage = read_csv("results/stage2/summary.csv")
    innovation = read_csv("results/innovations/summary.csv")
    heuristic = read_csv("results/innovations/heuristic_evaluation_summary.csv")
    extension = read_csv("results/extension/summary.csv")
    coverage = read_csv("results/extension/coverage.csv")
    cases = read_csv("results/extension/cases.csv")
    matrix = read_csv("results/stage2/distance_matrix.csv")
    if not any(r["status"] == "reused" for r in coverage) or not any(r["status"] == "measured" for r in coverage):
        raise ValueError("拓展覆盖索引缺少复用或新测记录")
    if not any(r["case"] == "Q1-01_corridor_25" for r in cases):
        raise ValueError("场景定义与绘图代码不一致")
    figures = [
        figure1(stage, output), figure2(stage, output), figure3(innovation, output),
        figure4(heuristic, output), figure5(extension, output),
        figure6(extension, innovation, output), figure7(extension, innovation, output),
        figure8(extension, output), figure10(matrix, output),
    ]
    doc = description(output)
    print("生成图表：", *(path.name for path in figures), sep="\n- ")
    print("说明：", doc)


if __name__ == "__main__":
    main()
