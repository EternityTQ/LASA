# Modern baseline 准备与协议审查（2026-09-16）

本次只准备代码、runner 和静态评估。没有启动真实训练、服务器任务、多 seed 或第二数据集正式实验。没有修改 CAGE 方法、R/Radial/NSGA-II、任何 defense 或 LASA。

## 进度核查与修改文件

开始时工作区已经存在 PoisonedFL 初稿、注册/CLI/engine 接口、FMNIST 配置和 cache 文件名修改；本次是在这些未提交修改上补齐，不是从零重做。`algorithms/attack/mos.py`、component ablation runner 及其已有测试的未提交修改均为先前工作，本次未编辑。

| 文件 | 本次任务中的作用 |
| --- | --- |
| `algorithms/attack/poisonedfl.py` | 独立攻击；核对并修正官方归一化公式，补充来源与差异说明 |
| `algorithms/attack/__init__.py` | 已有的 PoisonedFL 注册，核查保留 |
| `main.py` | 已有的 attack choice、scale/interval CLI，核查保留 |
| `algorithms/engine/fedavg_all.py` | 已有显式状态接口；修正被 finite-audit 跳过时的状态回滚快照 |
| `config/attack/fmnist/basee.yaml` | 保留已对齐的本地参数，显式写出 `cnnfmnist` |
| `utils/data_pre_process.py` | FMNIST v2 cache、内容校验、确定性 split、原子写入；合并已有重复分支 |
| `run_mos_baselines_server.sh` | 复用 cell 生命周期；加入数据集/协议、PoisonedFL 参数、FMNIST manifest/summary |
| `run_fmnist_modern_baselines_server.sh` | 新的 4×3 矩阵入口、单 seed、两轮 smoke、setting 校验 |
| `tests/test_poisonedfl.py` | 9 项无训练测试 |
| `tests/test_fmnist_split_cache.py` | 2 项 synthetic split 测试 |
| `tests/test_run_fmnist_modern_baselines.sh` | 模拟子进程验证 runner，无模型训练 |
| `MODERN_BASELINES_PREPARATION.md` | 本报告 |

## PoisonedFL：官方对应关系

审查固定到官方 commit [`266488e2cbe5953aab61712f315518546f457e55`](https://github.com/xyq7/PoisonedFL/tree/266488e2cbe5953aab61712f315518546f457e55)。对照了 `byzantine.py`、`test_agr.py`、`nd_aggregation.py` 和 `scripts/cifar.py`、`scripts/FashionMNIST.py`。不是基于第三方 baseline 实现。

以本轮开始全局参数为 W_t，上轮接受的全局变化为 h_t，上一恶意向量为 g，固定随机符号为 s：

| 官方步骤 | 本地实现与检查 |
| --- | --- |
| 一次采样固定符号 | `sign(randn)`，每个 `fedavg_all()` 独立状态；PyTorch RNG，不要求与 MXNet 随机数逐位相同 |
| 首轮 warm-up | `test_agr.py` 先放零 fake updates，攻击在没有 history 时不覆盖；本地将本轮恶意槽位的可训练参数清零 |
| 观察历史 | h_t = W_t − W_(t−1)，使用实际接受的模型变化，不读取 benign update 来构造方向 |
| 残差幅度 | a = abs(h_t − g·norm(h_t)/(norm(g)+1e−9))；单列向量逐行 norm 等价于 abs |
| 符号与归一化 | v = a·s/(norm(a)+1e−9)；去掉初稿的均匀方向回退，零残差保持零 |
| 反馈降幅 | 默认每 50 轮判断符号对齐数；不足 k99 且 0.7c≥0.5 时 c←0.7c |
| 攻击向量 | c·norm(h_t)·v；复制到当前实际恶意槽位，客户端总数不变 |
| checkpoint 时点 | 与官方一样使用 e=0,50,… **聚合后的**模型作下一段基线；本地通过下一轮入口延迟捕获，避免偏移一轮 |
| last_grad | 当本轮存在恶意槽位时保存攻击向量；零恶意轮保留上一向量 |

攻击使用 `local_model − global_model` 的 model delta；官方也是加法 model delta，因此没有额外翻转符号或再乘 local learning rate。

明确的 adaptation 差异：

1. MXNet → PyTorch；官方增加 fake clients，本地使用当前统一 compromised-client protocol，固定 100 个客户端中的 20 个受控、每轮均匀无放回抽 25 个，只替换其中实际选中的受控槽位。不会添加 fake clients，也不会强行凑足 5 个。官方 240 fake 相对 1,200 genuine 为 20%，相对合计 1,440 为 16.67%；本地 20% 的分母是总池 100，因此也不能把两个比例直接视为相同。
2. 只向 `named_parameters()` 中 `requires_grad` 且 floating 的条目注入攻击。冻结参数、浮点 BN running statistics、整型 `num_batches_tracked` 均保留该客户端本地 update。官方无对应的本地 ResNet18 BN buffer 语义。这只规定攻击输出，不改变既有 defense 对 state_dict 的处理。
3. 官方 k99 只列出四种维度；本地使用 `ceil(d/2 + Φ⁻¹(0.99)·sqrt(d/4))` 大维度近似。四个官方维度测试相差最多 1；不宣称是精确二项分位数，亦不宣称小维度统计检验等价。当前两个模型均为百万维。
4. 所有状态归属于当前 experiment；没有模块全局攻击状态。零恶意轮仍观察全局模型；在尚无任何恶意参与时 `last_malicious=0`，补足官方 `last_grad=None` 的未定义边界。随机 sign 恰为零时置为 +1。
5. 沿用当前 engine 的非有限值拒绝/回滚，状态与模型同步回滚；不导入官方聚合包装中的 NaN/Inf 替换或 norm-defense 原地修改。若最初轮次被拒绝，首次接受的攻击调用仍 warm-up。异常拒绝后的轨迹属于 framework adaptation，应记录 rollback。
6. CIFAR/FMNIST 默认攻击参数沿用官方脚本 `sf=8`，反馈间隔 50、decay 0.7、停止降幅条件 0.5。架构、客户端数、IID、tau、batch size、训练轮数沿用当前统一实验设置，不复刻官方 CNN、1,200 genuine + 240 fake、local_epoch=1 的整体训练方案。

这些差异同时在模块注释和 `[PoisonedFL]` 输出标明。

必须维护的跨轮状态：`fixed_sign`、`scale_factor`、`previous_global`、`feedback_checkpoint`、`checkpoint_after_round`、`last_malicious`、`last_round`，以及状态兼容性信息 `trainable_keys`、`dimension`、`feedback_interval`。checkpoint 与 previous_global 分工不同，不能合并。未来如增加模型断点续训，必须连同这些张量和 RNG/optimizer 状态一起存取；当前 runner 的 resume 是 **cell 重试**，失败 cell 从 round 0 和新攻击状态重新开始。

正式 runner 应记录：源码 commit/本地源码 hash、dataset/model、seed/repeat、client pool/participation/恶意比例、每轮实际恶意数、split 路径与 SHA256、攻击参数初值 8/50/0.7/0.5、k99 规则与参数维度、warm-up/buffer 策略、每轮当前 c 与 history_norm、反馈轮对齐计数/k99/降幅前后 c、rollback、defense budget。已将静态参数写入 command/environment，动态参数写入 train.log；FMNIST root summary 记录 dataset/protocol/model/恶意比例/defense budget 和 PoisonedFL 的 scale/interval。

## Multi-Krum / Trimmed Mean budget 审查

**建议采用 B：事先固定的 server Byzantine budget；不把每轮真实恶意数直接传给 defense。当前矩阵继续明确标记 ff=10，以保持已有 CIFAR 可比性；这是一项协议建议，没有修改 defense。**

事实依据：当前 `byzantine_robust_aggregation.py` 两个函数默认 `n_attackers=10`，engine 的单 defense、多 defense 两条调用路径均省略此参数。官方上游 [`8477367a4e8708cde264f7572805040c650af59f`](https://github.com/JiiahaoXU/LASA/tree/8477367a4e8708cde264f7572805040c650af59f) 同样如此；上游 CIFAR 也是 100 用户、25 参与，main 将 20% 换算为期望 5 后，engine 再用于确定受控客户端池。**固定 10 是继承的原始代码行为，不是本次新增 PoisonedFL 漏传。** 但源码/配置没有说明为何取 10，不能由此证明原论文有意定义了这个 upper-bound protocol；应称为“继承的未显式参数化 budget”。

标准假设：[Krum 原论文](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) 假定至多 f 个 Byzantine，条件 n>2f+2；[Trimmed Mean 分析](https://proceedings.mlr.press/v80/yin18a/yin18a.pdf) 允许 trimming fraction 为恶意比例的上界，要求保留有效样本。它们不要求 server 知道真实坏客户端身份或每轮真实数量。实验模拟器能计算真值，不代表现实 server 能访问这个 oracle。

| 设定 | 评价 |
| --- | --- |
| A：f_t=当轮真实恶意数 | oracle-count defense，会随攻击采样改变防御强度；只适合作为显式标注的额外实验，不宜主协议 |
| B：固定 ff | clean/所有 attacks/两个 dataset 共享同一 budget，server 不接触恶意真值；推荐 |
| C：ff=ceil(0.2×25)=5 | 可以称 nominal contamination budget，但 5 是期望，不能称本抽样协议的真实 upper bound |
| C：固定每轮 5 个恶意客户端 | 可使 ff=5 成为真正上界，但改变客户端抽样 threat model，影响所有已有实验，本次不实施 |

当前 K_t~Hypergeometric(N=100,K=20,n=25)：E[K_t]=5，P(K_t>5)=37.6467%，P(K_t>10)=0.1229%。若各轮独立抽样，200 轮至少一次超过 10 的概率约 21.81%。所以 **ff=10 也仅是事先配置的保守 budget，不是保证成立的最坏情况上界**；全池 20 人均可能被抽中，而 ff=20 又不满足 n=25 的 Krum 条件。保留当前 threat model 就应把结果描述为经验评估，报告超预算轮数，不能声称全程满足经典鲁棒性定理。

ff 对行为影响很大：当前 iterative Multi-Krum 在 n=25,ff=10 时选 3 个候选；ff=5 时选 13 个。Trimmed Mean 每端去 10 个，只剩 5 个；ff=5 时剩 15 个。更小 ff 不保证精度/攻击强度单调变化。

还发现原始 Krum score 的排序切片包含自身零距离，实际只累计 n−f−3 个其他邻居，和标准 n−f−2 个其他邻居有差异；迭代删除式 multi selection 也要作为实现细节披露。该问题独立于 budget，本次未修正。若后续改成标准实现，应统一重跑对应 defense，不能只换标签。

clean 使用相同 ff=10。`non_attack` 保留相同抽样/名义受控客户端池，但实际注入量为零，不应因已知实验是 clean 就把 defense budget 改成 0。当前 `args.malicious_attackers_this_round` 在 non_attack 清零之前被赋值，因此统计时须区分“被标记受控的选中槽位数”和“有效注入数”；不可把 clean 的该诊断字段解释为实际攻击数。

已有 CIFAR 是否重跑：

- 若全矩阵实际都使用当前 ff=10 和相同实现，结果仍是这个协议下有效的经验结果，**无需仅因期望恶意数是 5 就全部重跑**；冻结时补齐协议标注与命令/代码版本核查。
- 若将 ff 改为 5、oracle f_t，或修正 Krum score/selection，则 Multi-Krum、Trimmed Mean 对应 clean 和所有 attack cells 必须按受影响范围统一重跑；不可混用新旧预算结果。
- 若改变客户端抽样，全部 defenses 的相关结果都受影响。SignGuard 不会仅因 MK/TM 的 budget 参数调整而自动失效。
- 服务器上的实际完成情况与历史日志未远程核验，因此这里是重跑判定条件，不是对每个既有 CIFAR cell 的认证。

论文最推荐的统一落地方式：保持当前随机参与协议，把 ff=10 显式命名、预先固定且对 clean/全部 attack 共用；真实 K_t 仅供离线审计。未来批准参数化后使用独立 `defense_f`，与 CLI 的全池攻击百分比 `num_attackers` 分离。若论文必须要求逐轮 worst-case 上界，则需要重新设计抽样协议，不能靠把默认值换成 5 达成。

## FMNIST setting 与 runner

Fashion-MNIST + CNNFmnist；100 用户、25 参与、20% 固定受控池、IID；200 轮、tau=3、batch=64、local_lr=0.1、local_momentum=0.9、decay=0.99、global_momentum=0.9、clip=2.0。后者与当前 CIFAR 配置对齐；保留现有 normalization。实际 learning-rate decay 时点继承 engine。

v2 split cache 将 dataset、实际 IID/non-IID 模式、用户数、数据长度和 seed 编入名字；旧 `fmnist_dict_users.pik` 和先前只有用户数的名字均不会读取。不删除旧 cache。检查用户 key、非空、索引整数/范围/唯一性及 IID 样本数；不合法的新 cache 明确报错，避免静默换实验分割。`freeze_datasplit=0` 不读 cache；FMNIST runner 显式使用 1。生成 split 时保存/恢复 NumPy RNG，命中与未命中不会改变后续客户端选择。原子 replace 避免写出半个缓存。

默认矩阵仅为 `non_attack,agrAgnosticMinMax,poisonedfl_attack,mos_attack` × `multi_krum,tr_mean,signguard`。CAGE flags 沿用现有 runner：strict、dual、adaptive init=1、boundary_only=0、radial=1；正式结构冻结时应再次核对。

`ATTACKS`/`DEFENSES` 或 `ONLY_ATTACK`/`ONLY_DEFENSE` 可筛选单个或子矩阵。裸 `SMOKE=1` 默认只跑 `poisonedfl_attack × signguard` 的 2-round cell；显式给出筛选值时运行指定子矩阵，`SMOKE_ROUNDS` 可覆盖轮数。

默认输出：`server_experiments/modern_baselines/fmnist_cnn_u100_s25_m20_iid_splitv2_legacyff10/<timestamp>/`。smoke 多一层 `/smoke/`。每 cell 有独立 config 和工作目录，attempt 下保留 command/environment/train.log/status/heartbeat/metrics/summary，cell 顶层同步最新 attempt。顶层 `summary.csv` 保留全部已有 cells，单 cell resume 不丢其他行。

resume 会跳过已完成且轮数匹配的 cell；失败产生新的 `attempt_N`。`protocol.txt` 校验参数与 config/生产源码 SHA256，拒绝跨 setting 或源码不一致的复用。没有模型级中途续训；不要同时用多个 launcher 写同一个 RESUME_DIR。代码或协议改变后使用新输出目录。environment 中保存源码 hash、Git 状态、参数和 Python 可执行文件，运行库差异仍需服务器启动前核对。

服务器 smoke 命令（已准备，**未执行**；在可用环境和 GPU 上显式运行）：

```bash
# 无训练验证
python -m unittest discover -s tests -p 'test_poisonedfl.py' -v
python -m unittest discover -s tests -p 'test_fmnist_split_cache.py' -v
bash tests/test_run_fmnist_modern_baselines.sh

# 两轮：round 0 warm-up，round 1 首个非零反馈攻击
GPU=0 SMOKE=1 ATTACKS=poisonedfl_attack DEFENSES=multi_krum \
  bash run_fmnist_modern_baselines_server.sh
GPU=0 SMOKE=1 ATTACKS=poisonedfl_attack DEFENSES=tr_mean \
  bash run_fmnist_modern_baselines_server.sh
GPU=0 SMOKE=1 ATTACKS=poisonedfl_attack DEFENSES=signguard \
  bash run_fmnist_modern_baselines_server.sh

# 与上述两轮目录匹配的 cell-level resume
RESUME_DIR=/absolute/path/to/smoke/timestamp GPU=0 SMOKE=1 \
  ATTACKS=poisonedfl_attack DEFENSES=multi_krum \
  bash run_fmnist_modern_baselines_server.sh

# CIFAR PoisonedFL 短 smoke，另建目录；同样未执行
GPU=0 ROUNDS=2 SEEDS=1 ATTACKS=poisonedfl_attack DEFENSES=multi_krum \
  bash run_mos_baselines_server.sh
```

两轮 smoke 不会覆盖默认第 50 轮降幅，该逻辑由 synthetic interval=2 单元测试覆盖；如服务器要核查动态降幅，可另用 `ROUNDS=3 POISONEDFL_FEEDBACK_INTERVAL=2`，并明确它是非默认测试参数。

## HiDRA 静态可行性（未实现）

依据作者官方 [`sarthak-choudhary/HIDRA@31174d52f8c4839966a01ef2e9b7b327035d51e1`](https://github.com/sarthak-choudhary/HIDRA/tree/31174d52f8c4839966a01ef2e9b7b327035d51e1)，检查 `src/attack.py`、`src/simulate.py`、requirements 和 train.sh，并对照[论文 Algorithm 5/多 chunk 讨论](https://www.comp.nus.edu.sg/~prateeks/papers/Hidra.pdf)。

兼容性中等偏好：官方训练也是 PyTorch，攻击核心却是 NumPy CPU 数组；requirements 为 torch 1.13.1、torchvision 0.14.1，模拟器还设置全局 float64。可以单独移植公式到设备上的 torch tensor，不能直接导入整个 simulate.py。需做 state_dict 展平/还原、trainable-only、实际恶意 slots、k=0，以及官方 `global−local` 与当前 `local−global` 的符号约定核对。HiDRA 核心没有 PoisonedFL 那样的跨轮反馈状态。

threshold 不是可忽略的参数：核心使用 `(sqrt(20)−1)·threshold` 来算攻击幅度；官方 `sigma=1e−5` 是固定方差尺度，adaptive 模式逐参数张量以 chunk=1000 做 covariance/eigh。Filtering/No-Regret 自然有这种方差 threshold；MK/TM/SignGuard 没有同义参数，不能把 ff 或 SignGuard 的 norm/sign 阈值代入。

可以在这三种 defense 上运行同一预先冻结的 HiDRA 向量生成规则，作为迁移攻击经验对比；但不能声称是各自最优的 defense-adaptive HiDRA 或官方已验证适配。官方 README 列出 Krum/TM 支持，提供的 train.sh 主要跑 average/filterl2/ex_noregret；没有 SignGuard 实现。逐 defense 另调 sigma 会引入额外 tuning budget，必须让其他攻击拥有一致的调参机会并披露。

信息预算也需要处理：官方 full knowledge 的 mean 使用所有攻击前 updates；partial variant 的 mean 只用受控客户端，但 simulate.py 的 adaptive threshold 仍由所有 updates 估计。当前 `agrAgnosticMinMax` 只读取受控槽位，不能默认把官方 full knowledge/adaptive threshold 当成同信息协议。若采用 partial 且限制 threshold 也由受控样本估计，这又是应明确记录的 adaptation。

静态开销估计（没有跑训练或 HiDRA benchmark；下列维度仅实例化本地模型计数）：

| 本地模型 | 浮点可训练维度 d | 官方逐 tensor、1000 维 chunk 数 | 25×d float64 数组本身 |
| --- | ---: | ---: | ---: |
| CIFAR ResNet18 | 11,173,962 | 11,214 | 2.081 GiB |
| FMNIST CNNFmnist | 1,663,370 | 1,669 | 0.310 GiB |

固定 threshold 版本主要是均值/方向/覆盖，约 O(nd)，torch 实现可避免 CPU 往返；不必做特征分解。官方 adaptive 版本每个 chunk 构造 b×b covariance 并 eigh，约 O(ndb + db²)，b=1000；CIFAR 每轮约 1.12 万次、FMNIST 约 1,669 次最多 1000×1000 特征分解，可能成为主耗时。上述内存不含原始 updates、模型和中间副本，不能当作总峰值。n=25 时可改用样本 Gram 矩阵/SVD 显著降成本，但必须验证数值/threshold 一致性；不宜直接照搬官方 CPU eigh 流程。无法仅静态推断具体秒数。

与 PoisonedFL 相比：固定 sigma 的 HiDRA 核心代码更短且无需跨轮状态；公平 threshold/information protocol 的设计和 adaptive 性能验证反而更贵。PoisonedFL 的反馈主要 O(d)，长期状态约 4 个 d 维向量，且无 covariance/eigh。

**结论：不建议现在把 HiDRA 定为此三-defense 矩阵的正式第二个现代 baseline。** 原因是目标防御不具备其自然方差 threshold、partial 信息限制需要再适配、adaptive 实现开销和公平调参成本较高；不是“不能移植”。当前优先完成 PoisonedFL 更清楚。若后续确定采用，最小范围是一个独立 attack 文件、注册和 CLI threshold/chunk/knowledge 参数、统一 slots/dtype 接口、公式/threshold 测试与 runner 元数据；无需修改 CAGE/defense，也不应引入整套官方 engine。当前没有新增 HiDRA 代码或其他 baseline。

## 本地验证结果与边界

- Python 静态编译、`git diff --check`、两个 runner 的 Bash 语法检查通过。
- PoisonedFL 9 项 CPU 单元测试通过，包括独立 NumPy 公式对照和一次 synthetic server update；未使用数据、optimizer 或本地训练。
- split cache 2 项 synthetic 测试通过；真实 `load_partition`/`iid` 函数执行，dataset/序列化边界替换为测试对象，不下载 Fashion-MNIST。
- FMNIST mock runner 测试通过：产物、6 个模拟 cells、completed skip、单 cell resume 保留完整 summary、协议不匹配拒绝、失败重试。模拟进程只输出预设日志。
- 原有 `tests/test_run_mos_baselines.sh` 模拟编排回归通过（含已有 component runner 的 mock 检查）；没有运行 CAGE 训练。
- 本地隔离测试环境为 `.dist/prep-venv`，Python 3.14 / PyTorch 2.14.0+cpu / NumPy 2.5.2；没有验证服务器 CUDA 环境、真实数据集加载或整条训练 engine。正式运行前仍应执行上面的服务器 smoke。

正式实验仍等待 component ablation 完成、CAGE 结构和 defense protocol 冻结。
