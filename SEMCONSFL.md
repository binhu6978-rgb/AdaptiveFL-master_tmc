# SemConsFL 第一阶段实现

在当前 AdaptiveFL 框架内接入，关注 Full 模型准确率；不实现 CRD、知识蒸馏、额外原型损失或新的选择器。

## 接入位置

- `models/SemConsFL.py`：私有 adapter/分类器、共享冻结测量头、原型提取、语义记忆、方向门控和增量聚合。
- `Algorithm/Training_SemConsFL.py`：沿用原训练循环的模型构建、客户端选择、异构反馈、参数切片及七模型测试。
- `main_fed_ori.py`：默认进入 SemConsFL；显式 `--algorithm AdaptiveFL` 仍进入原基线。修复原日志文件名引用不存在的 `agg_exp`。
- `models/__init__.py`：补齐原算法导入所需的 `ResNet18_widar` 导出，不改变模型。

原 `models/Fed.py`、`models/Update.py`、ResNet、数据划分、客户端资源设定及测试代码均未修改。

## 固定条件与实现约定

ResNet18（本仓库版本）、CIFAR-10、非 IID Dirichlet α=0.3。默认保留 100 客户端、每轮 10 个选择槽位、5 本地 epoch、batch 50、1000 轮、SGD lr=0.01、衰减 0.998、momentum=0.5、weight decay=1e-4、available 选择和 4:3:3 资源比例。不增加数据增强，不改 `drop_last=True`。少于一批的客户端仍不执行优化步骤。

必须区分论文机制与以下明确的工程解释：

- 本地目标为 task CE + semantic CE，两者均更新 backbone；不向客户端增加 anchor loss。共享测量头冻结但允许梯度穿过。
- 测量空间 128 维，adapter 输出维度取 Full classifier 输入维度。测量头固定正交初始化，adapter 为零补齐单位映射，私有分类器零初始化。
- adapter 按 `(client_id, feature_dim)` 保存，语义分类器按 client 保存，仅在本次进程内持久化，不上传聚合。
- 原型来自本地训练后的额外无梯度遍历，类样本数至少 5；遍历隔离随机数状态，不改变后续训练/选择的随机序列。
- 当前原型先与上一轮 anchor 比较，再更新 anchor；窗口严格为最近 5 个通信轮，不是最近 5 次参与。
- 保留原选择器重复抽到同一客户端的行为。所有训练槽位照常聚合；同一客户端每轮只有一条证据，重复类原型加权合并且样本计数不重复累加；方向参考取该客户端覆盖最大的槽位，Q 按有效槽位计算。
- 关键层按实际输出通道异构比排序，平局按参数量、原顺序排序，选前三个卷积权重。方向参考至少需要 3 个不同客户端；仅对实际覆盖坐标计算中位数。偶数中位数取中间两值均值，MAD 不乘校正系数。
- 增量相对实际下发参数计算。保留原框架首轮七个模型独立初始化的行为；这是一项与统一全局初始化不同的兼容性约定，并不声称增量相减能消除该差异。
- 聚合权重为所有槽位样本量归一化后的 `w_i * gamma_i`，不按坐标覆盖量重新归一化；仅当 `0 < gamma_bar < 0.05` 时保护性放大到 0.05，零更新保持不变。

## 运行

在项目根目录使用已安装 PyTorch 的 Python 环境：

```powershell
& 'D:\software\minicoonda\envs\fd\python.exe' -B -m unittest discover -s tests -p test_semconsfl.py -v
& 'D:\software\minicoonda\envs\fd\python.exe' -B -u main_fed_ori.py --algorithm SemConsFL --epochs 3
```

正式训练将 `--epochs 3` 改为 `--epochs 1000`，不改变其他基础参数。数据根目录可用已有的 `SEMCONSFL_CIFAR_ROOT` 环境变量指定；应指向完整、可读的数据缓存。工作区联调缓存位于 `tmp/semconsfl_smoke/cifar_cache`。

对照基线显式运行 `--algorithm AdaptiveFL --iid 0 --epochs 1000`，不能遗漏 `--iid 0`：原解析器默认为 IID。保持同 seed 和所有其余参数，核对实际划分索引，而不只看 α 文件名。原数据缓存文件名可能含 `_bal`，实际划分实现强制 unbalanced；本实现不更改这一既有行为。

## 输出和判断

每次运行输出到当前工作目录 `result_semconsfl/<时间戳>/`：

- `config.json`、`partition.json`：实参、工程约定和实际划分 SHA256。
- `rounds.jsonl`：客户端/模型选择、local loss/steps、证据/可靠度/β、方向统计、γ、保护性更新、七模型准确率及 Full 准确率。
- `accuracy.txt`、`test_time.txt`：原文本布局，按运行隔离，避免同日结果追加混合；原 `get_final_acc` 报告逻辑保留。
- `summary.json`：完整七模型曲线、末轮及最佳 Full；最佳 Full 仅作补充，不替代原评价。
- `full_final.pt`：末轮 Full 参数，仅用于测试，不是完整恢复训练检查点。

短程联调只验证数据流和数值稳定性。3 轮准确率接近随机水平不能说明最终优劣。100 客户端低参与率配合 5 轮历史窗口会使 β 下限和可信集合回退频繁触发，需要观察完整曲线及门控日志；不要为提高结果擅自延长窗口、增加预算或更换基线评价方式。

## 已完成验证（2026-09-12）

11 项单元测试通过；两次完整 3 轮 GPU 联调通过，每轮均测试全部七个模型和完整测试集。实际划分与现有 seed=1、α=0.3 缓存逐客户端逐索引一致，总计 50,000 样本，最小客户端 17 样本。两次运行的选择、模型反馈、样本量和本地优化步数一致。

最新联调目录为 `tmp/semconsfl_smoke/result_semconsfl/20260912_171611`，Full 为 10.59%、10.83%、10.89%；七行准确率文本、JSON 汇总和含 65 个 state 项的有限数值 checkpoint 均已验证。前次 Full 为 10.60%、10.88%、10.85%。原框架只设置种子、未强制确定性算法，因此不承诺 GPU 逐位复现；本实现没有更改这一全局设置。未启动 1000 轮正式实验，也未验证优于 AdaptiveFL。
