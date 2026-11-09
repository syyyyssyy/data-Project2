# 实验二：搜索算法

项目实现四道题：迷宫 A*、加权地形 A*、博弈树 Minimax/Alpha-Beta、竞争宝藏分配。内容按**基础算法与正确性验证、优化算法与对照实验、自设场景与适用性分析**组织；四道题的具体分析见[实验报告](EXPERIMENT_REPORT.md)，报告逐题说明基础实现、改进方法及场景结果。本 README 汇总运行方法、文件关系和数据来源。

## 环境配置

需要 **Python 3.10+**。四题核心算法和 `unittest` 只用标准库；运行实验脚本中的绘图以及 `plot_results.py`、`plot_analysis.py` 需要 `matplotlib` 和 `numpy`。在项目目录执行：

```powershell
cd experiment2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

在 macOS/Linux 中，将激活命令换为 `source .venv/bin/activate`。若只运行 `main.py` 与核心测试，可以不安装绘图依赖。检查环境可运行 `python --version` 和 `python -c "import matplotlib, numpy; print(matplotlib.__version__, numpy.__version__)"`。脚本会使用系统可用的中文字体；若生成的图片出现方框字，需要安装微软雅黑、思源黑体或黑体之一后重新运行绘图脚本。

## 快速开始

在当前目录运行：

```powershell
python main.py --task all
python -m unittest discover -s tests -v
python plot_results.py
python plot_analysis.py
```

第一条输出四题 PDF 样例；每个 `Task ... output:` 下一行是题目要求的简洁结果，`Task ... stats:` 另列节点统计。第二条运行当前 **36 项**自动化测试。第三条只读取已有 CSV，重建图1～图8、图10；第四条在固定场景上重新计算逐格启发值和逐层访问数，并从既有估值 CSV 绘制图11～图13。两条绘图命令都**不改写原始实验数据或重新计时**。只看某题时用 `--task 1`、`2`、`3` 或 `4`。PDF 样例的结果摘要见下文“PDF 样例核对”。

需要重新测量时再运行以下命令；它们会覆盖**各自目录**中的实验 CSV 与旧图片，计时受本机负载影响：

```powershell
python stage2.py --repeats 20 --seed 2026
python innovation_experiments.py --repeats 50
python extension_experiments.py --repeats 20
python plot_results.py
python plot_analysis.py
```

一次只运行一个实验批次，以免并行任务干扰计时。三批实验结果分别保存在 `results/stage2/`、`results/innovations/`、`results/extension/`；完整数据索引见下文。

## 项目文件说明

阅读解题代码建议按 `main.py → astar.py → minimax.py → treasure_game.py` 的顺序；对照实验代码可之后再看。

| 文件 | 作用 | 主要入口与需要关注的内容 |
| --- | --- | --- |
| `main.py` | 四道题的命令行入口，加载 PDF 样例并打印路径、代价、树值、距离矩阵及宝藏决策。 | `run_task()` 按题号调用算法；`main()` 解析命令行参数。 |
| `astar.py` | 题一和题二共用的四方向 A* 主循环；检查地图、计算启发值、维护优先队列、回溯路径。题四计算距离时也调用它。 | `search()` 是核心；`maze_path()` 和 `weighted_cost()` 分别对应 PDF 的两种直接输出；`path_cost()` 独立复算路径。默认 `heuristic="auto"` 使用经过论证的下界。 |
| `minimax.py` | 题三的树节点、Minimax、Alpha-Beta 和搜索统计。 | `alpha_beta_score(root)` 直接返回 PDF 要求的整数得分；`alpha_beta()`/`minimax()` 返回带统计的结果对象；`pdf_tree()` 构造 PDF 样例。 |
| `treasure_game.py` | 题四的 7×7 地图、宝藏价值、六点距离矩阵、不可变博弈状态、单次行动与对抗搜索。 | `best_treasure_choice(matrix, H)` 直接返回 PDF 样例字典；`solve()` 返回带统计的结果对象；`distance_matrix()` 求六点矩阵。 |
| `experiments.py` | 记录四题要求样例和固定种子自设输入的初步对照，保存逐次测量与时间中位数。 | `run_comparisons()`、`save_csv()`；不参与四题的最终解题逻辑。 |
| `stage2.py` | 基础搜索策略对照：固定输入和随机种子，重复计时，核验答案一致，记录逐次测量及均值，再从 CSV 绘图。 | `run()` 负责测量，`charts()` 负责绘图；`generated_tree()` 生成固定种子的完整博弈树。 |
| `innovations.py` | 四题五种独立拓展：A* 同分优先深入、加权行列下界、Alpha-Beta 置换表、宝藏状态记忆化、深度截止时的未来收益估值。 | `maze_deeper_first()`、`weighted_axis_lower_bound()`、`alpha_beta_transposition()`、`treasure_memoized()`、`treasure_estimated()`；详见根目录的 `INNOVATIONS.md`。 |
| `innovation_experiments.py` | 独立创新对照，不覆盖基础搜索策略对照数据。 | `python innovation_experiments.py --repeats 50`；输出 `results/innovations/`。 |
| `extension_experiments.py` | 拓展场景：单通路、交错地形、固定树重排、六/八宝藏；复用旧数据的索引及新数据绘图。 | `python extension_experiments.py --repeats 20`；输出 `results/extension/`。 |
| `plot_results.py` | 只读取现有 CSV 与原场景构造代码，绘制统一风格的报告图；不重新运行搜索。 | `python plot_results.py`；输出 `results/final_figures/`。 |
| `plot_analysis.py` | 补绘题二启发值与真实代价、题三逐层剪枝、题四深度决策与收益损失；逐格和逐层数据另存 CSV，并核对旧实验统计。 | `python plot_analysis.py`；输出图11～图13、`analysis_data/` 和补充说明。 |
| `requirements.txt` | 列出实验绘图所需的 `matplotlib`、`numpy`；四题核心代码仅用标准库。 | 使用 `python -m pip install -r requirements.txt` 安装。 |
| `tests/test_astar.py` | 测试 PDF 的迷宫与加权样例、非法输入、无解、等长路径、免费终点和路径复算。 | `python -m unittest discover -s tests -v`。 |
| `tests/test_minimax.py` | 测试 PDF 树、叶节点、非法树、剪枝及访问顺序。 | 同上。 |
| `tests/test_treasure.py` | 核对地图与宝藏、A* 距离矩阵、收益和状态转移、H=2/H=4 结果。 | 同上。 |
| `tests/test_experiments.py` | 检查基础对照实验的固定种子和采样记录。 | 同上。 |
| `tests/test_stage2.py` | 检查完整树的可复现性、排序前后最优值及基础搜索策略对照 CSV 分组。 | 同上。 |
| `tests/test_innovations.py` | 检查前四种拓展的答案一致性，以及第五种估值的截止条件、完整深度结果、决策质量和独立 CSV。 | 同上。 |
| `tests/test_extension.py` | 检查拓展场景、输出索引及六类边界情况。 | 同上。 |
| `tests/test_output_contract.py` | 检查四题的简洁返回类型，以及命令行结果是否单独成行。 | 同上。 |
| `results/` | 保存基础实验实测 CSV 和距离矩阵。 | `raw_runs.csv` 为逐次记录，`median_times.csv` 为时间中位数，`distance_matrix.csv` 为六点距离。 |
| `results/stage2/` | 保存基础搜索策略对照结果。 | `raw.csv` 为 640 条逐次测量，`summary.csv` 为 32 个配置的平均时间和确定性指标，`distance_matrix.csv` 为六点距离，五张 PNG 来自 `summary.csv`。 |
| `results/innovations/` | 独立的四题五种创新数据。 | 前四种用 `raw.csv`、`summary.csv` 与四张 PNG；估值拓展另用 `heuristic_evaluation_raw.csv`、`heuristic_evaluation_summary.csv` 和 `task4_heuristic_evaluation.png`。 |
| `results/extension/` | 拓展实验的新测记录和旧数据索引。 | `cases.csv` 定义场景，`coverage.csv` 指向来源，`raw.csv`/`summary.csv` 记录 27 组、540 次新测，四张 PNG 从 CSV 绘制。 |
| `results/final_figures/` | 供实验报告选择使用的统一风格图表与说明。 | 12 张 240 dpi PNG；`图表说明.md` 与 `图11-13_说明.md` 记录数据来源、指标与限制。 |

`__pycache__/` 中的 `.pyc` 是 Python 运行时自动生成的缓存，不是需要阅读或修改的源代码。解题时先使用默认配置：题一、二 `search(..., heuristic="auto")`，题三 `alpha_beta(pdf_tree())`，题四 `solve(distance_matrix(), H)`。`h=0`、Minimax、逆序和其他排序参数主要供对照实验使用；节点排序可能减少访问数，但不能保证每个输入都更快。

## 单题运行与 Python 接口

在 `experiment2` 目录执行：

```powershell
python main.py --task 1
python main.py --task 2
python main.py --task 3
python main.py --task 4
python main.py --task all
python -m unittest discover -s tests -v
```

题三现在也可直接使用 PDF 第 10–11 页的 JSON 结构：将完整样例保存为 `tree.json`，运行 `python main.py --task 3 --tree-json tree.json`。在 Python 中，`json.load()` 得到的字典可直接传给 `alpha_beta(data)` 或 `minimax(data)`，用 `.value` 读取最佳整数分数；原有 `TreeNode` 输入仍可使用。JSON 内部节点需有 `is_maximizing_player` 与非空 `children`，根为 `true`，后续 MAX/MIN 交替；叶子只需整数 `value`。

四题命令行均先打印 `Task ... output:`，下一行单独打印 PDF 要求的结果；`Task ... stats:` 另列统计。题四的结果行用 JSON 双引号输出。若从 Python 调用并只需简洁返回值：

```python
from astar import maze_path, weighted_cost
from minimax import alpha_beta_score
from treasure_game import best_treasure_choice, distance_matrix

path = maze_path((0, 0), (3, 4), maze_grid)          # 坐标列表；无解为 []
cost = weighted_cost((0, 0), (4, 4), weighted_grid)  # 整数；无解为 -1
score = alpha_beta_score(tree_root_or_json)          # 整数，例如 5
choice = best_treasure_choice(distance_matrix(), 2) # {'best_target': 'T2', 'best_value': -1}
```

导出距离矩阵及运行对照实验：

```powershell
python main.py --task 4 --export-matrix
python main.py --task all --experiments --repeats 5 --seed 42
```

第二条命令覆盖 `results/raw_runs.csv` 和 `results/median_times.csv`。随机地图使用局部 `random.Random(seed)`，地图可复现；运行时间受机器负载影响，不能要求每次完全相同。原始 CSV 保留每次测量，汇总 CSV 只对同输入、同算法、同启发函数、同深度和访问顺序的时间取中位数。脚本只写实际测量数据。

## 规则与接口

`search(grid, start, goal, weighted=False, free_goal=False, heuristic="auto")` 返回 `SearchResult`，包含路径、代价、扩展数、生成数、耗时。`maze_path(start, goal, grid)` 直接返回题一要求的坐标列表；`weighted_cost(start, goal, grid)` 直接返回题二要求的整数代价或无解时 `-1`。无解的 `search` 路径为空、代价为无穷大。非法坐标、障碍起终点、不规则地图和非法加权地形抛出 `ValueError`。

A* 的 OPEN 是优先队列；`g` 为已知最小代价，`f=g+h`。仅在找到更低 `g` 时更新父节点并入队；弹出时丢弃过期条目。搜索节点数表示找到终点前实际取出并检查的地图位置数（包括终点）；生成数表示放入待搜索队列的次数，同一位置若找到更低代价可能再次入队。普通迷宫使用曼哈顿距离。加权图的自动启发是 `最小进入权值 × 曼哈顿距离`；若终点免费进入，则改为 `最小进入权值 × max(曼哈顿距离 - 1, 0)`。该下界要求所有非障碍进入代价为正且仅最后进入目标的动作免费；本实现强制正整数地形，其他规则可选 `heuristic="zero"`。`path_cost` 独立逐步验算路径。自定义启发函数由调用者负责保证可采纳性。

题三根节点为 MAX，子节点依层交替，默认从左到右。`TreeNode` 可以是有整数 `value` 且无子节点的叶子，或无 `value` 且至少有一个子节点的内部节点；空叶、带值内部节点、环和非节点子项均视为非法。PDF 中共享的 `max2` 子树允许被两处引用，每次到访分别计数。`alpha_beta_score(root)` 直接返回整数；需要统计时，`minimax` 与 `alpha_beta` 返回含 `.value` 的 `TreeResult`。`order` 回调可重排子节点；Alpha-Beta 仅在 `alpha >= beta` 时剪枝。访问节点数表示实际计算的博弈局面数；剪枝数表示直接跳过的候选分支数，一个分支下面可能还有许多未计算的局面。

题四的六个关键点均按进入该格代价 1 处理，因此 A* 的两两最小代价对称；每对都实际进行正反向搜索并验证。若不连通，矩阵元素为无穷大，`take` 会明确拒绝无法到达的选择。`GameState` 不可变；选中宝藏才领取，经过其他宝藏不领取；已领取宝藏仍可通行。深度 `H` 是剩余行动次数。并列最优时选当前遍历顺序中的第一个目标；默认顺序为 T1、T2、T3、T4。`best_treasure_choice(matrix, H)` 返回只有 `best_target`、`best_value` 的字典；`solve(matrix, H, algorithm="alpha_beta")` 返回含统计的对象。叶节点评估数包含深度截止或宝藏取完的终止状态；访问数包含根节点；耗时只测搜索本身，距离矩阵构建另行进行。

## 样例核对

| 题目 | 预期 | 实际运行 |
| --- | --- | --- |
| 1 迷宫 | 最短路径 7 步 | 7 步 |
| 2 加权地形 | 最小代价 7 | 7 |
| 3 博弈树 | 最优值 5 | 两种算法均为 5 |
| 4，H=2 | T2，收益差 -1 | 两种算法均为 T2，-1 |
| 4，H=4 | T3，收益差 3 | 两种算法均为 T3，3 |

PDF 题二明确规定进入 E 的最后一步免费；题四明确规定宝藏格进入代价为 1，本项目分别实现。PDF 题三的树形示意与文字回传描述略有不一致，测试和实现以 PDF 第 10–12 页的具体树结构及预期根值 5 为准。

## 基础搜索策略对照：实验设计与性能评估

在本目录运行 `python stage2.py --repeats 20 --seed 2026`。需预先安装 `matplotlib`；该命令重跑实验，覆盖 `results/stage2/` 中的 CSV 和 PNG。`stage2.py` 会直接验证所有对照策略在同一输入上的最终值一致，然后用 `summary.csv` 生成图表。`raw.csv` 包含每次实测值、题号、实验编号、参数、种子、重复编号、机器名与 Python 版本；`summary.csv` 使用同组运行时间的算术平均值。每个配置重复 20 次，路径和节点统计必须在各次重复中一致。32 个配置共 640 条原始记录。每次实验仅在同一运行环境内比较；微秒级耗时受调度噪声影响，不能仅凭这些小差异断言速度优势。

对照输入与策略：题一为 PDF 5×5 迷宫及种子 2026 生成的 20×20 迷宫，比较 `h=0` 和曼哈顿距离；题二为 PDF 加权地图、`test_astar.py` 中的两张自设小地图，以及 `generated_weighted_grid()` 生成的 20×20、30×30、40×40 和目标位于内部的 30×30 地图，比较 `h=0` 和 `最小进入权值 × max(曼哈顿距离-1,0)`。四张大地图分别使用种子 2026–2029；权值为正整数，障碍概率与最大权值见 `stage2.py`。生成器保留顶行和目标列的可达走廊，所有案例均采用进入 E 免费规则。题三为 PDF 树及深度 6、每层分支 4 的完整树，叶值由 `random.Random(2026)` 按从左到右顺序在 [-100,100] 均匀取整数；比较 Minimax、默认 Alpha-Beta 和“首叶估计”排序的 Alpha-Beta。首叶估计在 MAX 节点按各子树最左叶值从高到低排，在 MIN 节点从低到高排；仅用于排序，不改变评价值，其排序开销包含在计时内。题四比较 H=2/H=4 的 Minimax、默认/逆序/即时收益降序 Alpha-Beta。即时收益为当前玩家从当前位置获取目标的 `宝藏价值-最短路代价`；并列仍保留 T1–T4 原顺序。节点访问数含根与叶；剪枝数为跳过的直接子分支数，不能与跳过的全部后代数混用。

五张图分别是 `task1_expanded.png`、`task2_expanded.png`、`task3_visited.png`、`task3_pruned.png`、`task4_depths.png`，均从真实 `summary.csv` 读取。题二的 PDF 地图从 15 降至 11；两张小地图和 20×20 地图持平；30×30 角落目标从 672 降至 609，40×40 从 1436 降至 1418，30×30 内部目标从 318 降至 194。节点减少不保证时间减少。题三完整树显示首叶估计排序虽减少叶节点评估（742→724），却增加总访问数（1168→1272）和平均耗时；其局部剪枝数不代表总工作量。题四即时收益排序在 H=4 把访问数从 47 降至 37；是否更快应以当前 CSV 的计时和更充分的复测判断。

## 优化算法与对照实验：四题五种创新方法

五种可选方法集中在 [`innovations.py`](innovations.py)，由 [`innovation_experiments.py`](innovation_experiments.py) 在同一场景内与原方法比较。原题仍通过 `main.py` 调用标准实现；创新方法没有替换其默认行为。`results/innovations/raw.csv` 为逐次调用记录，`summary.csv` 为按题目、场景、算法和深度汇总的结果。前四种方法共 **29 组 × 50 次 = 1450 条**；估值方法另有 **18 组 × 50 次 = 900 条**，单独保存在 `heuristic_evaluation_raw.csv` 和 `heuristic_evaluation_summary.csv`。随机迷宫与共享 DAG 的种子为 2026，随机加权地图为 2027；重复运行同一固定输入，不能理解为 50 个独立随机实例。计时用外层 `perf_counter()` 测完整函数调用，包含预处理及缓存构建；只应与**本批次同场景**的控制组比较。

| 题目 | 创新方法与接口 | 对照设计及原理 | 适用条件与限制 |
| --- | --- | --- | --- |
| 一 | `maze_deeper_first()`：A* 的 `f` 相同时优先较大 `g` | 与默认 FIFO 比较；仅改变同分节点的弹出顺序，曼哈顿下界与最短路径规则不变。 | 大量候选节点 `f` 相同且有多条路线时可能有效；单通路或小图可能持平。 |
| 二 | `weighted_axis_lower_bound()`：必经行列下界，加同分深入 | 三组为全图最小代价下界、只改同分顺序、行列下界加同分深入。跨越目标行和列之间的每条必经边界，至少要付该行或列可进入格的最低费用；最后进入 E 免费，只扣一次终点费用，再截断为非负值。 | 地形最低值呈空间差异时下界可能更紧；整行/列不可通行时安全退化为 0。预处理和估值可能比省下的搜索更贵。 |
| 三 | `alpha_beta_transposition()`：精确值置换表 | 与普通 Alpha-Beta 比较；键包含节点对象身份及 MAX/MIN 身份。只有子树全部被精确求值才缓存，剪枝所得界限不能当精确值复用。 | 同一 `TreeNode` 被多个父节点引用的共享 DAG 可受益；没有共享状态的普通树无法靠缓存减少求值。 |
| 四 | `treasure_memoized()`：未剪枝 Minimax 状态记忆化 | 与同递归公式的 `cache=False` 比较。键为双方位置、剩余宝藏、行动方、剩余步数；缓存未来收益差，历史累计分数在根处合入。 | 重复局面多时可减少求值；小规模时字典开销可能抵消收益。此方法**没有 Alpha-Beta 剪枝**，不能把其节点数与原 Alpha-Beta 的剪枝效果混为一谈。 |
| 四 | `treasure_estimated()`：深度截止未来收益估值 | H=0 且仍有宝藏时，以当前收益差加确定性贪心模拟后续取宝收益差；与同实现的原评价 `use_estimate=False` 对照。 | 属于深度受限的近似评价，可能改善也可能误导首步；宝藏取完时仍使用精确收益差。 |

下表的“节点”在题一、二指算法找到终点前实际取出并检查的地图位置数，在题三、四指实际计算的博弈局面数；两类数量只在同一道题、同一场景内比较。时间是每组固定输入重复 50 次的平均值。题二的“原方法 → 创新方法”同时改变下界与同分顺序，另设控制组区分影响。结果见 [`summary.csv`](results/innovations/summary.csv)。

| 场景 | 最终结果 | 原方法 → 创新方法：节点 | 平均时间（ms） | 对照发现 |
| --- | --- | ---: | ---: | --- |
| 题一 PDF 迷宫 | 7 步 | 14 → 14 | 0.031 → 0.032 | 小图没有扩展优势。 |
| 题一空旷 25×25 | 48 步 | 625 → 49 | 1.429 → 0.150 | 同 `f` 平局很多，深入策略明显减少扩展。 |
| 题一随机 20×20 | 38 步 | 235 → 47 | 0.537 → 0.137 | 固定种子 2026 的图也受益。 |
| 题二 PDF 加权图 | 代价 7 | 11 → 9 | 0.051 → 0.069 | 只改同分顺序的控制组也是 9；行列预处理使总时间增加。 |
| 题二条纹 25×25 | 代价 71 | 601 → 49 | 2.192 → 0.657 | 只改同分顺序为 481，行列下界提供主要节点收益。 |
| 题二随机 30×30 | 代价 215 | 609 → 597 | 2.014 → 2.398 | 节点略减，但完整调用更慢。 |
| 题三 PDF 树 | 根值 5 | 14 → 13 | 0.019 → 0.013 | 仅 1 次缓存命中；时间太短，不宜推断普遍速度。 |
| 题三共享 DAG | 根值 -10 | 770 → 129 | 0.786 → 0.222 | 缓存命中 159 次；共享子树带来明显收益。 |
| 题四 PDF，H=4 | T3／3 | 65 → 53 | 0.065 → 0.075 | 命中 12 次，缓存版仍略慢；原 Alpha-Beta 只访问 47。 |
| 题四六宝藏，H=6 | T1／6 | 1957 → 487 | 1.989 → 1.193 | 命中 510 次，较大状态空间中缓存更有价值。 |

估值方法需用**完整搜索事后评价固定首步**，不能把截止时的 `best_value` 当实际收益。PDF 四宝藏 H=3 时，原评价选 T2，完整收益差 1、相对完整最优值损失 2；估值选 T3，完整收益差 3、损失 0。六宝藏 H=1 时情况相反：原评价选 T1，完整收益差 6、损失 0；估值选 T2，完整收益差 5、损失 1。PDF H=4、六宝藏 H=6 已取完全部宝藏，估值没有调用。数据见 [`heuristic_evaluation_summary.csv`](results/innovations/heuristic_evaluation_summary.csv)。四张节点图与一张决策质量图保存在 [`results/innovations/`](results/innovations/)；更详细的推导、场景坐标和图表解读见 [INNOVATIONS.md](../INNOVATIONS.md)。

## 自设场景与适用性分析

拓展批次研究地图结构、地形分布、访问顺序和问题规模的适用条件。运行 `python extension_experiments.py --repeats 20` 会覆盖 `results/extension/` 中本批次的 CSV 和图，不改动前两批数据。输入均固定；只有题三叶值由种子 **2026** 生成。**27 组 × 20 次 = 540 条**新测记录。计时包含完整算法调用，不含场景构造、CSV 写入与事后校验。复用旧数据时只跨批次比较确定性的节点数，不拼接运行时间。

[`cases.csv`](results/extension/cases.csv) 定义场景；[`coverage.csv`](results/extension/coverage.csv) 按场景、策略和深度指向来源，状态为 `measured`（本批次实测）、`reused`（引用创新 CSV）、`test_only`（仅正确性测试）。[`raw.csv`](results/extension/raw.csv) 每次调用一行，[`summary.csv`](results/extension/summary.csv) 汇总同配置结果、平均与中位耗时。覆盖表中有 **27 个 measured、5 个 reused、6 个 test_only**；没有出现在新测 CSV 中的组合不一定未运行。四张拓展图分别为[题一结构图](results/extension/task1_structure.png)、[题二地形图](results/extension/task2_terrain.png)、[题三访问顺序图](results/extension/task3_order.png)和[题四规模图](results/extension/task4_growth.png)。

### 题一：单通路与开放地图

两图均为 25×25，起点 `(0,0)`、终点 `(24,24)`，在各自地图内比较 FIFO 与同 `f` 时较大 `g` 优先。新测蛇形单通路最短 **336 步**，两者均扩展 **337** 个节点；复用创新批次的空旷图最短 **48 步**，扩展数 **625 → 49**。单通路几乎没有路线可选，空旷图有大量同 `f` 候选。由于两图最短路径长度不同，不能把 337 和 625 的绝对值直接归因于岔路数量。

### 题二：条纹与高低交错地形

两图均为 25×25，进入 E 免费。旧条纹图每第 5 列权值为 7，其余为 1；新棋盘图的权值 1 与 9 交错。下表后三列是**扩展节点数**。

| 场景 | 最优代价 | 全图最低费下界 | 加同分深入 | 行列下界＋同分深入 |
| --- | ---: | ---: | ---: | ---: |
| 条纹图（复用创新 CSV） | 71 | 601 | 481 | 49 |
| 交错图（拓展新测） | 239 | 625 | 625 | 625 |

交错图另测 `h=0`，也是 **625** 个节点、代价 **239**；条纹图没有测 `h=0`。交错图每行每列都有权值 1 的格子，行列最低费难以识别必经的高价路径。交错图的 `h=0`、全图下界、加同分深入、行列下界平均调用时间依次为 **1.819、2.136、2.139、2.551 ms**。此图中预处理没有换来节点收益，单张图也不能证明所有交错地形都如此。

### 题三：固定树的访问顺序

深度 6、分支数 4 的完全树有 **4096 个叶子、5461 个总节点**；种子 2026 固定叶值。三种顺序的根值均为 **-36**，Minimax 均访问 **5461** 个节点。原顺序不重排；有利顺序在 MAX 层把真实值高的子树排前、MIN 层把真实值低的排前；不利顺序反向。

| Alpha-Beta 顺序 | 访问节点 | 访问叶子 | 跳过的直接子分支 | 平均时间（ms） |
| --- | ---: | ---: | ---: | ---: |
| 原顺序 | 1168 | 742 | 537 | 1.310 |
| 事后有利 | 268 | 127 | 297 | 0.336 |
| 事后不利 | 4806 | 3530 | 299 | 4.316 |

表中的“跳过的直接子分支”只计当前博弈局面不再考虑的候选行动，不是这些行动下面所有未计算的局面；总体工作量应结合实际计算的局面数判断。有利／不利顺序依据**子树真实 Minimax 值事后构造**，构造成本未计入搜索时间，不能视作现成排序器。真正可计算的“首叶估计”排序在同一棵树的基础对照中使访问数 **1168 → 1272**，因此不能把事后有利顺序的结果归给它。

### 题四：宝藏数量与搜索深度

四宝藏使用 PDF 7×7 地图；六、八宝藏使用确定性的**无障碍 8×8 抽象距离矩阵**，不是在 PDF 地图上单纯加宝藏。六宝藏的 A=`(0,0)`、B=`(7,7)`，T1–T6 坐标依次为 `(0,4)`、`(2,1)`、`(2,6)`、`(5,2)`、`(6,5)`、`(7,0)`，价值依次为 9、6、7、7、5、8；八宝藏另加 T7=`(3,3)`／10、T8=`(4,6)`／4。`solve(..., values=...)` 和 `take(..., values=...)` 的可选价值表仅用于扩展场景；省略时仍按 PDF 原值运行。

| 场景 | H | 首步／收益差 | Minimax 访问 | 默认 Alpha-Beta 访问 | 即时收益排序 Alpha-Beta 访问 |
| --- | ---: | --- | ---: | ---: | ---: |
| PDF 四宝藏 | 4 | T3／3 | 65 | 47 | 37 |
| 网格六宝藏 | 4 | T1／3 | 517 | 130 | 83 |
| 网格六宝藏 | 6（完整） | T1／6 | 1957 | 362 | 243 |
| 网格八宝藏 | 4 | T7／4 | 2081 | 531 | 232 |
| 网格八宝藏 | 8（完整） | T7／5 | 109601 | 16864 | 2592 |

三种算法在**同一场景及深度**的收益差一致。八宝藏完整深度下，默认 Alpha-Beta 平均 **85.335 ms**，即时收益排序平均 **16.558 ms**；排序利用即时收益 `宝藏价值－当前位置到宝藏距离`，有额外计算成本，在此固定输入仍明显减少总开销。H=4 与完整深度不能混作相同决策质量的比较；不同宝藏数也不是同一地图的纯规模变化。

### 边界情况与结论范围

[`tests/test_extension.py`](tests/test_extension.py) 验证六类情况：普通迷宫不可达、加权终点不可达、多个等长最短路、所有宝藏即时收益为负、并列首步、宝藏提前取完；同时检查选不可达宝藏抛出 `ValueError`。这些**只做正确性测试，没有重复计时**，不在 `raw.csv` 中充当性能样本。本轮结论适用于列出的固定场景；尚缺多布局、多随机种子的分布实验。详细构造与局限见 [EXTENSION.md](../EXTENSION.md)。

## 报告图表

运行 `python plot_results.py` 会从 `results/stage2/`、`results/innovations/`、`results/extension/` 的现有 CSV 生成图1～图8及图10，保存在 [results/final_figures/](results/final_figures/)。运行 `python plot_analysis.py` 会额外生成[图11](results/final_figures/图11_加权启发函数紧度检验.png)、[图12](results/final_figures/图12_博弈树逐层剪枝效果.png)和[图13](results/final_figures/图13_宝藏深度决策与收益损失.png)，并将新分析数据保存到 `analysis_data/`。[原图说明](results/final_figures/图表说明.md)和[补充图说明](results/final_figures/图11-13_说明.md)分别列出来源、指标与限制。图9需要完整决策树搜索轨迹，现有 CSV 只有汇总统计，因此没有凭空画出。绘图不覆盖原有实验 CSV；跨批次只合看确定性的节点数，时间比较只在同批次内进行。

## 当前完成状态

四题标准算法、PDF 样例校验、36 项自动化测试、基础搜索策略对照、五种独立创新方法、拓展场景与边界测试均已实现并留有实际运行数据。[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) 已是包含图表的长篇报告草稿；若正式提交，还需按课程页数和格式要求取舍排版。上述结论仅适用于列出的场景与计时条件；题三事后真值排序不是现成可用的实战排序器，题四深度截止估值也不保证完整搜索的最优决策。



