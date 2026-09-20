# owm_memory — 物体中心世界模型（LPWM / C-JEPA）在 RoboMME 上的记忆测试

实现依据：`/common/home/zl1308/Projects/Object-centric-new/oc_wm_robomme_final.md`（下称"规格"）。
本文档说明：① 环境；② 每一步训练/测试怎么跑；③ 规格第 17 节 16 个「⚠️ 需核实」的核实结果；④ 与规格的偏差及原因；⑤ 本次已实际跑通/未跑的部分。

- 代码：`/common/home/zl1308/Projects/owm_memory`（只放代码和配置）
- 大文件（帧缓存、生成的测试集、checkpoint、证据缓存、日志）：`/common/home/zl1308/data/owm_memory_data/{cache,data,outputs}`
  （`~/data` → `/common/users/zl1308`；`/common/home` 有 208 GB 配额，135 GB 的帧缓存放不下）。路径在 `configs/experiment.yaml → paths`。
- 第三方仓库**一行未改**：`Projects/cjepa`、`Projects/lpwm`、`Projects/robomme_benchmark`。

---

## 0. 环境

```bash
cd /common/home/zl1308/Projects/owm_memory
source scripts/env.sh          # 固定解释器 = conda env memory_owm，并重置 PYTHONPATH
python -m pytest -q tests     # 40 个单元测试（无需数据/GPU）
```

`scripts/env.sh` 会**覆盖** `PYTHONPATH`：登录 shell 的 `PYTHONPATH` 里有 `condBFNPol_latest`，它的 `utils/` 会遮蔽 LPWM 的 `utils/`，导致 `No module named 'utils.util_func'`。

`memory_owm` 原先缺少 C-JEPA/分析所需的包，我已用 **constraints 文件锁住全部已装包的版本**后补装（未改动任何已有包）：
`pandas pyarrow openai pytest pytorch_lightning omegaconf torchmetrics webdataset transformers timm seaborn tensorboard`。
`torchcodec` 的 wheel 与当前 torch 2.6+cu126 ABI 不兼容，已卸载；VideoSAUR 只在 import 时需要它，`owm/shims/torchcodec` 提供一个占位模块（我们的 shard 存 `video.npy`，不解码 mp4）。

GPU：5 × RTX 6000 Ada 46 GB。下文命令用 `CUDA_VISIBLE_DEVICES=...` 指定。

---

## 1. 目录

```
configs/experiment.yaml         全部固定变量（规格 16.2）+ 路径
configs/preregistration.yaml    预注册（P3：看任何测试结果前 git commit）
configs/decision_rules.yaml     决策规则文字版（实现：owm/data/decision_rules.py）
configs/option_vocab.yaml       各任务选项（P1 自动生成）
configs/videosaur_robomme.yml   VideoSAUR 配置（由官方 pusht_dinov2_hf.yml 派生）
configs/lpwm_robomme.json       LPWM 256px 配置（由官方 bridge.json 派生）；lpwm_robomme_128.json = 官方 128px
owm/data/      h5_reader sampling splits goal_parser decision_rules decision_index gt_state key_events
owm/wm/        videosaur_utils cjepa_predictor_ext train_cjepa_predictor cjepa_adapter cjepa_action
               lpwm_dataset lpwm_train lpwm_adapter actions evidence future_perturb
owm/readout/   model dataset train evaluate
owm/analysis/  bootstrap verdict tables
owm/baselines/ astra_prompts astra_client astra_offline astra_closed_loop
owm/closed_loop/runner.py
scripts/       p0 … p8 入口脚本（见下）
tests/
```

包名用 `owm` 而不是规格里的 `src`：C-JEPA 仓库自身的顶层包就叫 `src`，会冲突。

---

## 2. 逐步运行

每个阶段结束有关卡（规格第 15 节）。耗时为本机实测或估计。

### P0 帧缓存（一次性，约 15–60 min，~135 GB）
```bash
python scripts/p0_extract_cache.py --source train --tasks all --workers 16
```
把每条 episode 的 `front_rgb` 全部帧 + 逐帧标签从 h5 导出到 `cache/episodes/{source}/{task}/episode_i/{frames.npy, meta.npz}`；之后任何阶段都不再读 h5（h5 在 NFS 上按 timestep 分组，读取很慢）。

### P1 标签与规则
```bash
python scripts/p1_labels.py --timelines 5      # 时间线、选项是否固定、目标解析覆盖率、视频段动作是否有信息、决策索引
python scripts/p1_key_events.py                # 关键事件检查：outputs/key_events/<task>/episode_i.jpg + summary.txt
# 方案 A：用内置规划器为官方 val seed 生成离线测试集（6 任务 × 50，6 个进程约 1 h，~45 GB）
python scripts/p1_generate_test_split.py --gpus 2 3 4 --workers-per-gpu 2
#   可选：每任务再加 200 条新 seed（规格 5.2 样本量）： --extra 200 --seed-base 2000000
python scripts/p0_extract_cache.py --source test --tasks eval --workers 12
```
关卡：规则覆盖全部 episode（每条 ≥1 个记忆关键决策 ✔）、目标解析 100% ✔、关键事件可见 ✔（见第 3 节）。
方案 B（70/30）：`experiment.yaml` 里 `split_plan: B`，不需要生成。

### P2 真值（重放，600 条约 40–80 min）
```bash
# 只需要这一条：重放 train + test，最后自动重建 decisions.parquet 并写 outputs/reports/p2_gt.txt
python scripts/p2_gt.py --gpus 2 3 4 --workers-per-gpu 3
# 中途挂了直接重跑同一条命令即可（已有 data/gt/*.npz 的 episode 会跳过）

# 仅当真值不变、但索引规则变了时用（future_offsets / aug_offsets / decision_rules.py / split_seed / 可见性阈值），几十秒： optional
python scripts/p2_gt.py --report-only
```
用 `setup/seed` 重放 joint_action，逐帧读取模拟器里所有物体位姿 + 前视分割图（可见性），存 `data/gt/...npz`，并把标签点匹配到 `target_obj`。
关卡：重放一致、匹配失败 < 2% ✔；同时给出 n_max → `cjepa.num_slots = n_max + 3`。

### P3 预注册
```bash
git add -A && git commit -m "preregistration"      # 提交 configs/preregistration.yaml（填上 registered_on）
```

### P4 参照读出头（下限 / 上限；每个头 3–6 min，可按任务并行，见第 6 节）
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/p4_train_readout.py --conditions floor ceiling            # 评测 test
CUDA_VISIBLE_DEVICES=1 python scripts/p4_train_readout.py --conditions floor ceiling --eval-split val   # 不碰 test 的自检
```
关卡：上限 ≥ 95%、下限 ≈ 机会水平（结果见第 5 节）。

### P5 C-JEPA
```bash
# 5a  webdataset shards（video.npy，stride-16 序列；2 个 offset ≈ 19 GB）
python scripts/p5a_make_videosaur_shards.py --offsets 0 8
# 5b  用仓库自带的 VideoSAUR 训练器训练（252 px，N 个 slot，官方 100k 步）
CUDA_VISIBLE_DEVICES=2 bash scripts/p5b_train_videosaur.sh 20
# 5c  冻结 VideoSAUR，提取世界模型训练/验证 episode 的 slot（默认 16 个 offset，约 2–3 h；--offsets 0 4 8 12 更快）
CUDA_VISIBLE_DEVICES=2 python scripts/p5c_extract_slots.py
# 5d  预测器（W=context_frames, F=len(future_offsets)，物体级掩码 N//4，无动作/本体感知）
CUDA_VISIBLE_DEVICES=2 python -m owm.wm.train_cjepa_predictor --tag main
# 验收（规格 7.3）：验证损失平稳 + 分解/预测可视化
CUDA_VISIBLE_DEVICES=2 python scripts/p56_acceptance.py cjepa --episodes 20
```

### P6 LPWM
```bash
# 先探测显存能放下多少帧（batch 1, fp32）
CUDA_VISIBLE_DEVICES=4 bash scripts/p6_lpwm_memory_probe.sh configs/lpwm_robomme.json 11 19 27
# 训练（调用仓库自带 train_ddlp；数据集通过进程内替换 get_video_dataset 接入，不改仓库）
CUDA_VISIBLE_DEVICES=3 python -m owm.wm.lpwm_train                       # 单卡
CUDA_VISIBLE_DEVICES=2,3,4 accelerate launch --num_processes 3 -m owm.wm.lpwm_train --accelerate   # 多卡（可加 --mixed_precision bf16，未验证数值稳定性）
CUDA_VISIBLE_DEVICES=3 python scripts/p56_acceptance.py lpwm --episodes 20
```
`timestep_horizon` 由 `experiment.yaml` 的 `context_frames + 未来步数 − 1` 自动得到，保证两个模型上下文长度一致（**见第 4 节：80 帧在本机显存下训练不了 LPWM，需要两个模型同时缩短**）。

### P7 预测提取 + 未来扰动测试
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/p7_extract_evidence.py --model cjepa
CUDA_VISIBLE_DEVICES=3 python scripts/p7_extract_evidence.py --model lpwm        # 5 次采样，种子 0–4；--batched-samples 更快
```
每个决策（含训练集的伪决策，共约 1.1 万个）写 `cache/evidence/{cjepa|lpwm}/{source}/{task}/{episode}_{t}.npz`（规格 8.3 格式；C-JEPA 约 0.3 s/个，LPWM 5 次采样约 2–5 s/个），随后随机 100 个决策做未来扰动测试
（t 之后的帧全部换成噪声，整条流水线重跑，输出必须逐元素相同；前 5 个还做**反向对照**：把第 t 帧也换成噪声，输出必须变化）。有任何失败脚本以非零码退出。

### P8 离线评测与主表
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/p4_train_readout.py --conditions cjepa lpwm check_cjepa check_lpwm
python scripts/p8_tables.py            # outputs/tables/main_table.md|json + appendix.md（S、CI、预测检查、结论、是否需第 2 个 WM 种子）
```

### P9 GPT-6 Astra 离线（设置 A）
```bash
export OPENAI_API_KEY=...
python -m owm.baselines.astra_offline --dry-run --limit 3     # 不调用 API，只生成请求 + 标注图
python -m owm.baselines.astra_offline --limit 10              # 试跑：核对格式、坐标、单次费用
python -m owm.baselines.astra_offline                         # 全量（可断点续跑，不会重复计费）
python -m owm.baselines.astra_offline --frame-only            # 可选对照：只给决策帧
python scripts/p8_tables.py                                   # 主表自动多出 Astra 行
```

### P10 闭环（可选）
```bash
CUDA_VISIBLE_DEVICES=2 python -m owm.closed_loop.runner --condition floor     # 也可 cjepa / lpwm
python -m owm.baselines.astra_closed_loop                                     # 设置 B
```

### P11 动作实验（可选，规格第 14 节）
```bash
python scripts/p7_extract_evidence.py --model action_history                  # 泄露检查用的"动作历史"证据
CUDA_VISIBLE_DEVICES=2 python -m owm.wm.train_cjepa_predictor --actions --tag actions
CUDA_VISIBLE_DEVICES=3 python -m owm.wm.lpwm_train --actions
python scripts/p7_extract_evidence.py --model cjepa --actions                 # 扰动测试同时把 t 之后的动作换成噪声
python scripts/p7_extract_evidence.py --model lpwm --actions
python scripts/p4_train_readout.py --conditions action_history cjepa_act lpwm_act
python scripts/p8_tables.py --actions
```

---

## 3. 规格第 17 节「⚠️ 需核实」逐项结果

| # | 事项 | 结论（均已在本机实际核实，除注明者） |
|---|---|---|
| 1 | `choice_action` 何时有值；分界帧含义 | **每一帧都有值**（不是只在分界帧）：`{"choice":"A","point":[y,x]}`，label **大写**（选项是小写），point 每帧重新投影、随物体移动；视频段为 `""/[]`；**`need_parameter=false` 的选项也存了 point**。`is_subgoal_boundary` = 新子任务的**第一帧**。另有两类要排除的分界：视频任务第 0 帧（空 choice）、结尾 `simple_subgoal=="All tasks completed"` 的重复分界。**timestep/episode 都是 0 起始**（规格以为 1 起始）→ 帧索引 = timestep 编号，不需要 −1。h5 第 k 帧是执行完 action k **之后**的观测。视频段里（VideoRepick/MoveCube）也有 A/B 分界，属于演示，不算决策。 |
| 2 | 同一任务选项是否固定 | **固定**：16 个任务、每个 100 条，选项变体数都是 1 → label 头用固定类别。 |
| 3 | 各任务选项文字；MoveCube 对应 | 见 `configs/decision_rules.yaml`。**MoveCube 共 5 个选项且全部 `need_parameter=false`**：勾=a,b；推=c；抓放=d,e → 记忆关键 = 视频后第一个决策的 label ∈ {a,c,d}（训练集 35/34/31）。Unmask 系列有"两次拿取"的 episode（目标含两种颜色）→ 目标向量 = 第 1 颜色 + 第 2 颜色；第二次拿取标为 `memory_secondary`（不进主指标，也不能当预测检查）。 |
| 4 | 能否为 val seed / 新 seed 生成演示 | **可以**。仓库里生成脚本已被删，但 `tests/_shared/dataset_generation.py` 和 git 历史里有完整流程；场景是 (seed, difficulty) 的纯函数。`scripts/p1_generate_test_split.py` 已实际生成 6×50 = 300 条（0 失败），格式与训练 h5 相同。注意 `RobommeRecordWrapper` 只有 `save_video=True` 才写 timestep。 |
| 5 | 物体名称与位姿属性 | 统一从 `env.unwrapped.segmentation_id_map` 枚举（id>17 为场景物体），`actor.pose.p` → `[1,3]`；按钮/MoveCube 的 peg 是 articulation（按 link 合并）。VideoRepick 简单/中等难度的方块名字叫 `bin_i`（sic）且同色（颜色从渲染材质读）；MoveCube 有两个屏幕外的 decoy。实现：`owm/data/gt_state.py`。 |
| 6 | 重放是否确定性 | **是**。600 条训练 episode 全部 `replay_ok=1.0`；重放画面与 h5 的平均像素差 0.03–0.12（大部分帧逐像素相同）。训练集 episode 0–5 录制时注入了失败恢复，不影响重放。 |
| 7 | 前视相机能否输出分割 | **能**（`obs_mode="rgb+depth+segmentation"`，`include_maniskill_obs=True`）→ 可见性 = 分割像素数 ≥ 20，不需要深度后备方案。内外参是 OpenCV 约定；标签 point 与 `project(actor.pose.p)` 的误差 ≈ 0.5 px；匹配失败率 0–1.3%。 |
| 8 | VideoSAUR slot 是否严格因果 | **是**（读代码确认）：逐帧 DINOv2（MapOverTime）+ ScanOverTime，`slots_t = SA(Pred(slots_{t-1}), feat_t)`，无时间注意力/BatchNorm。适配层仍对每个决策**只编码截断到 t 的帧**，并由扰动测试端到端验证。注意 slot 初始化在 eval 下也采样噪声 → 每条序列固定 torch 种子。 |
| 9 | VideoSAUR 解码器能否单独解码预测 slot | **能**：`model.decoder.module(slots[B,S,D]) → masks[B,S,P]`，252 输入 → 18×18，取掩码加权质心。已跑通。 |
| 10 | C-JEPA 预测器能否 W=80 + 填充掩码 | 位置编码是长度 = W+F 的可学习表，**82×N token 可行**。但**没有 padding mask**，且官方 from-slot 训练器会丢弃所有短于 W+F 的视频——stride 16 下**每条 RoboMME episode 都短于 82 帧**（最长 ~70），官方训练器会得到 0 个样本。官方 `inference()` 本身支持更短历史（把历史**右对齐**到时间表末尾）。因此训练改成与官方推理一致：变长历史、右对齐、批内同长度、无填充 token（`owm/wm/cjepa_predictor_ext.py`，子类，参数/ state_dict 与官方完全相同）。另两点：官方 `get_mask_indices` 每次用固定种子重建 RNG → 被掩码的 slot 下标永远不变（默认保持官方行为，`mask_sampling: random` 可改）；`dim_head` 参数在官方代码里未被使用（实际 head dim = 128/16 = 8）。 |
| 11 | LPWM 只用先验 rollout | `sample_from_x(hist, cond_steps=len, use_all_ctx=False, deterministic=False)`：历史转移用后验（只看历史帧，因果），rollout 每步潜在动作来自 **prior head**。**唯一的泄露路径是 `use_all_ctx=True`**（仓库的动画/指标代码默认开）——适配层不用，且只传 ≤t 的帧。粒子确定性、随机性全部来自潜在动作先验 → 调用前紧挨着设种子。关键点 `z_pos` 是 (y,x)∈[-1,1]。 |
| 12 | LPWM 80 帧显存 | **放不下**。实测（batch 1，fp32，46 GB 卡）：256px/64 粒子：12 帧 19.0 GiB，20 帧 30.1 GiB（≈1.4 GiB/帧，上限约 30 帧）；128px：24 帧 20.4 GiB。82 帧在两种分辨率下都 OOM。另外仓库**没有 256px 配置**，bridge 是 128px；256px 必须改 CNN 通道设置才能建模（见第 4 节）。 |
| 13 | 动作条件配置项 | LPWM：`action_condition/action_dim`，动作 = `batch[1]`（经 AdaLN 进入 context 模块）；C-JEPA 官方做法是把动作/本体感知当作额外 slot token（`MaskedSlot_AP_Predictor`）。两者都已实现（`--actions`）。 |
| 14 | 视频段 `joint_action` 是否有意义 | VideoUnmask/VideoUnmaskSwap：视频段动作**恒定**（std=0，无信息）；VideoRepick、MoveCube：**有真实演示动作**（泄露风险高）；PickXtimes、ButtonUnmaskSwap 无视频。是否泄露以 14.4 的检查为准。 |
| 15 | `multi_choice` 接口 | 每次 `env.step({"choice","point":[y,x]})` 就是一次决策（没有"请求决策"标志），返回该技能执行期间的全部密集帧；`reset()` 内部播放条件视频并把视频帧放在 obs 列表里；point 取最近候选（无距离阈值）；非法 label / 缺 point → `info["status"]=="error"` 且 `obs is None`；真值从 `env.unwrapped` 实时读取（与离线重放同一份代码）。 |
| 16 | Astra SDK 写法与计费 | 对照已安装的 `openai 3.16` SDK 源码核实：Responses API、`reasoning={"effort":"high"}`、`text={"format":{"type":"json_schema",...,"strict":true}}`、`input_image` + base64 data URL；模型名 `gpt-6-astra` 在 SDK 模型列表里。**未能核实**：图像 token 计费方式（需联网/账号；代码按 API 返回的 usage × 配置里的单价记账，先 `--limit 10` 试跑对账）。没有 API key，Astra 两个脚本只做过 `--dry-run` 级别的检查。 |

关键事件检查（规格 5.5）：三个 Unmask 任务里彩色方块在第 0–31 帧可见 → stride 16 下任何 offset 都**恰好 2 个采样帧**（刚好满足"至少 2 帧"）；每次交换 50 步（≥3 帧）；VideoRepick/MoveCube 的演示 >100 步。所有任务按 stride 16 采样后最长 ~70 帧 < 80，不会触发"只保留最近 80 帧"。

---

## 4. 与规格的偏差（都有原因；请确认）

**A. 预测步数 K = 2 → 4（`temporal.future_offsets: [16, 32, 48, 64]`）—— 这是规格判读规则 1 自己规定的处理。**
用一个不需要学习的 oracle 规则检验"真实未来里有没有答案"（训练集记忆关键决策，规则 = 与目标同类型的候选里，离 t+K 时机械臂 3D 位置最近的那个；MoveCube = 离棍子更近→a，否则夹爪闭合→c，否则 d）：

| 任务 | t+32（K=2） | t+48 | t+64（K=4） |
|---|---|---|---|
| PickXtimes / VideoUnmask / VideoUnmaskSwap | 100% | 100% | 100% |
| ButtonUnmaskSwap | **65.4%** | 98.7% | 100% |
| VideoRepick | **79.3%** | 97.7% | 97.7% |
| MoveCube | **64.3%** | — | 94.0% |

按完按钮后机械臂要先收回、MoveCube 在 t+32 夹爪还没闭合 → 两步确实不够（读出头在 K=2 的上限实测为 94/93/85/83/50/74%）。
改动对所有条件、两个世界模型同时生效（C-JEPA 的 F、LPWM 的 rollout 步数、`timestep_horizon`、上限证据都从这一个配置项派生）。

**B. 读出头训练（规格原样上限不泛化，三处改动，全部可在 `experiment.yaml → readout` 里关掉）**
1. `fourier_freqs: 16 → 6`：16 个频带最高到 2^15·π，位置特征成了哈希，读出头直接背下 ~100 个训练布局。
2. `resample_ids: true`：训练时**每次抽样**重新随机化真值身份编号（同一样本内候选与真值未来保持一致；评测时固定）。
   编号若按 episode 固定，目标物体的编号就是完美的记忆钥匙。与规格"随机分配防止把编号当答案"的意图一致。
   消融（VideoUnmask / PickXtimes 上限，val）：规格原样 0.24 / 0.25；只降频带 0.29 / 0.09；只重随机编号 0.24 / 0.53；两者都做 **0.92 / 1.00**。
3. `aug_offsets: [4, 8, 12, 16]`：**只用于训练**的伪决策——每个训练决策之后、同一子任务内的 t+d 帧（`choice_action` 每帧都有标注，label 相同、point 跟随目标物体，所以标签是精确的）。
   训练样本 ~1.8k → ~8.7k。代价：世界模型条件要为这些帧也提取证据（训练 episode 的预测次数 ×5）。测试/验证决策不受影响。设为 `[]` 即回到规格行为。
   结果见第 5 节。

**C. 机会水平的两处细化（只影响判读规则 2"下限是否显著高于机会"）**
- 指针决策：1 / **与目标同类型**的候选数（选项文字已经限定了物体类型，按钮/目标盘不是真正的备选；ButtonUnmaskSwap 按 1/全部候选算会误判泄露）。
- label 决策：只看目标向量的最优预测器在测试集上的准确率（"训练集中给定目标向量下的多数类"）。PickXtimes 的要求次数本身就是先验（次数=1 → 放下后必然是"停止"），
  下限头正好达到这个值（val 0.725 = 0.725），这不是画面泄露。无目标变量的任务（MoveCube）退化为规格的多数类比例。

**D. 其余**
1. **C-JEPA 训练用变长右对齐历史**代替"前端填充 + 掩码"（见 ⚠️10）。W=80 超出论文测试范围，报告中需注明。
2. **`num_slots = 20`**（不是 12）：P2 实测 n_max = 17（VideoRepick 困难档 15 个方块 + 按钮 + 机械臂），规格规则 N = n_max + 3。
3. **LPWM 上下文长度**：80 帧训练不了（⚠️12）。规格的后备是"两个模型同时缩短"，**需要你决定**：
   (a) 256px：`context_frames ≈ 24`（`timestep_horizon = context + 4 − 1 = 27`，28 帧 ≈ 41 GiB）；(b) 128px 官方 bridge 设置：`context_frames ≈ 40–45`；(c) 多卡只分摊 batch、不省单样本显存；`--mixed_precision bf16` 可再探（未验证数值稳定性）。
   改 `experiment.yaml → temporal.context_frames` 一处即可，两个模型、Astra 的历史帧数都跟着变。各任务 stride-16 序列长度：均值 13–43 帧、最长 ~70 帧；
   Unmask 类任务"方块可见"只在最前面 2 个采样帧 → 上下文短于 episode 长度时关键事件会掉出窗口，报告里要写清楚窗口覆盖情况。
   256px 配置相对 bridge.json 的必要改动：`patch_size 32` + `n_kp_prior 64`（`n_kp_prior` 必须等于 (image/patch)²）、`anchor_s 0.125`、`bg_ch_mult` 6 级；无条件；`eval_im_metrics/ctx_for_eval` 关闭（后者用真值未来）。
4. **VideoSAUR 缩放用官方 transform 的 bicubic**（规格写 bilinear）：训练与提取一致更重要；`h_flip_prob 0.5` 保留官方值。
5. **预测检查**：VideoUnmask / VideoUnmaskSwap / ButtonUnmaskSwap / MoveCube **没有**非关键的指针决策（MoveCube 无任何需要参数的选项）→ 只能报"样本不足"，
   规则 3/4 在这四个任务上无法区分（表中给 `no_memory_signal_check_insufficient`）。只有 PickXtimes（测试集 50 个）和 VideoRepick（45 个）能做预测检查。
6. 决策帧排除视频段内的分界和结尾 "All tasks completed" 分界（⚠️1）。目标匹配失败（>20px）全部发生在 `memory_secondary`（第二次拿取时目标容器被机械臂挡住，可见像素 <20）；记忆关键决策 0 失败。
7. LPWM 证据 token 不含背景粒子；测试演示不注入 failure-recovery；Astra 保存的请求里图像以 (episode, 帧号) 引用而非 base64。
8. 读出头的预测按 `predictions/<task>__<cond>__<split>.parquet` 分文件保存，可以按任务在多张卡上并行训练（`--tasks`）。

---

## 5. 本次实际跑通 / 未跑的部分

**已在真实数据上完整跑完**
- P0：16 任务 × 100 条训练 episode + 300 条生成的测试 episode 的帧缓存（共 ~160 GB，在 `~/data/owm_memory_data/cache`）。
- P1：标签/规则检查、目标解析（PickXtimes 27 / Unmask 各 9 / VideoRepick 6 / MoveCube 2 种指令，100%）、关键事件检查、**方案 A 测试集生成 300/300 成功**。
- P2：训练 600 条 + 测试 300 条全部重放成功（`replay_ok = 1.0`，平均像素差 0.03–0.12），决策索引（含 6960 个仅训练用的伪决策）。
- P4：下限/上限读出头（6 任务 × 3 种子）+ 主表代码（bootstrap、S、判读）。结果见下表（`outputs/tables/main_table.md`）。
- 单元测试 40 个通过。

**只做了冒烟测试（代码路径全部执行过，但模型几乎没训练，数值无意义）**
- VideoSAUR：仓库自带训练器 + 我们的 shard/配置（252px、12 slot）跑了 60 步并存了 checkpoint。
- C-JEPA：slot 提取 → 预测器训练（含 `--actions` 版本）→ 适配层 → 证据缓存 → **未来扰动测试 6/6 通过（含反向对照）** → `cjepa` / `check_cjepa` 读出头训练与评测。
- LPWM：通过启动器调用仓库自带 `train_ddlp`（256px，T=5，1 个 epoch；`--actions` 版本同样跑通）→ 适配层（5 次先验采样确实互不相同）→ **未来扰动测试 4/4 通过**；动作版本的扰动测试（帧 + 动作都换噪声）两模型各 3/3 通过。
- 验收脚本 `p56_acceptance.py`（两个模型）、闭环 runner（下限条件，PickXtimes 2 条：1 成功 1 失败，符合"不知道次数"的预期）、Astra 离线 `--dry-run`（请求与候选标注图）。

**没有跑（需要你来跑；都是长时间训练或需要 API key）**
- VideoSAUR 完整训练（官方 100k 步）、slot 全量提取、C-JEPA 预测器完整训练；LPWM 完整训练（先按第 4 节 D.3 决定上下文长度/分辨率）。
- 全量 P7/P8（世界模型条件的读出头、预测检查、主表的 S 与结论）。
- GPT-6 Astra 的真实 API 调用（P9/P10 设置 B）：没有 key；SDK 参数已对照本机 `openai 3.16` 源码核实，计费方式未核实。
- 闭环全量评测（P10）、动作实验全量（P11）。

**P4 关卡结果（测试集 = 官方 val seed 生成的 50 条/任务；3 个读出头种子平均；K=4，含伪决策增强）**

| | PickXtimes | VideoUnmask | VideoUnmaskSwap | ButtonUnmaskSwap | VideoRepick | MoveCube |
|---|---|---|---|---|---|---|
| 机会水平 chance | 69.9 | 26.7 | 29.3 | 29.3 | 33.3 | 28.0 |
| 下限 floor: GT candidates, no future | 61.2 [56.7, 65.8] | 32.0 [20.7, 44.0] | 22.0 [14.0, 30.7] | 42.7 [31.3, 54.0] | 34.2 [25.4, 43.9] | 34.0 [25.3, 43.3] |
| 上限 ceiling: GT candidates + real future | 96.7 [93.6, 99.3] | 98.0 [94.7, 100.0] | 100.0 [100.0, 100.0] | 99.3 [98.0, 100.0] | 98.2 [95.6, 100.0] | 98.7 [96.0, 100.0] |

Accuracy in %, mean over 3 readout seeds, [95% CI] = paired bootstrap over episodes.
Excluded from this table (reading rule 1, see experiment.yaml -> stats.exclude): {'VideoRepick': ['hard']}
Prediction-check chance levels: PickXtimes 26.9, VideoRepick 26.4

按难度分层（未剔除任何层）与健全性检查（主读出头在非关键决策上）：

| task | cond | easy | medium | hard |
|---|---|---|---|---|
| PickXtimes | floor | 61.0 [54.9, 68.1] | 57.7 [50.0, 66.7] | 63.0 [54.5, 71.0] |
| PickXtimes | ceiling | 100.0 [100.0, 100.0] | 96.2 [88.5, 100.0] | 93.8 [87.3, 100.0] |
| VideoUnmask | floor | 39.7 [23.1, 57.7] | 2.8 [0.0, 8.3] | 44.4 [19.4, 69.4] |
| VideoUnmask | ceiling | 100.0 [100.0, 100.0] | 91.7 [77.8, 100.0] | 100.0 [100.0, 100.0] |
| VideoUnmaskSwap | floor | 23.1 [12.8, 34.6] | 22.2 [8.3, 38.9] | 19.4 [2.8, 38.9] |
| VideoUnmaskSwap | ceiling | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| ButtonUnmaskSwap | floor | 53.8 [38.5, 69.2] | 27.8 [11.1, 44.4] | 33.3 [11.1, 58.3] |
| ButtonUnmaskSwap | ceiling | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 97.2 [91.7, 100.0] |
| VideoRepick | floor | 33.3 [21.8, 44.9] | 36.1 [22.2, 50.0] | 2.8 [0.0, 8.3] |
| VideoRepick | ceiling | 97.4 [93.6, 100.0] | 100.0 [100.0, 100.0] | 13.9 [0.0, 33.3] |
| MoveCube | floor | 32.1 [19.2, 44.9] | 36.1 [16.7, 55.6] | 36.1 [19.4, 55.6] |
| MoveCube | ceiling | 97.4 [92.3, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

| task | floor (non-critical) | ceiling (non-critical) |
|---|---|---|
| PickXtimes | 94.0 [88.7, 98.0] | 96.0 [92.0, 99.3] |
| VideoUnmask | — | — |
| VideoUnmaskSwap | — | — |
| ButtonUnmaskSwap | — | — |
| VideoRepick | 44.4 [31.2, 57.7] | 100.0 [100.0, 100.0] |
| MoveCube | — | — |

读法：
- **上限全部 ≥ 95%**（96.7 / 98.0 / 100 / 99.3 / 98.2 / 98.7）→ 读出头有能力从"真实未来"读出答案；之后世界模型条件下的错误可以归因于世界模型的预测。
- **下限都在机会水平附近**，但有两点要留意：
  1. ButtonUnmaskSwap 下限 42.7 [31.3, 54.0]，CI 下界略高于机会 29.3（主要来自 easy：53.8%）→ 按预注册的规则 2 会被判为"当前画面泄露"。
     可能原因：物理交换后容器的静止位置带有轻微痕迹。建议用 `--extra 200` 扩大测试集后再定；判读代码会如实输出 `floor_leak`。
  2. VideoUnmask medium 下限 2.8%（远**低于**机会 20%）：训练集里目标永远是 `bin_0`，下限头学到了与"生成顺序"相关的位置线索，在 5 个容器的布局上系统性地反向。
     S 以下限为基准，所以不影响 S 的定义，但说明"真值 3D 位置"本身带有少量与记忆无关的线索。
- PickXtimes 的机会水平 69.9% 是"只看要求次数"的最优先验（见第 4 节 C），下限 61.2% 不高于它。
- 这张表里还没有世界模型行：C-JEPA / LPWM 需要先完成 P5–P7 的完整训练。

读出头消融（上限，测试集，逐步加入改动）：

| 设置 | PickXtimes | VideoUnmask | VideoUnmaskSwap | ButtonUnmaskSwap | VideoRepick | MoveCube |
|---|---|---|---|---|---|---|
| 规格原样（16 频带、编号固定、K=2）— val | 0.25 | 0.24 | — | — | — | — |
| 6 频带 + 重随机编号，K=2 | 0.94 | 0.93 | 0.85 | 0.83 | 0.50 | 0.74 |
| 同上，K=4 | 0.92 | 0.99 | — | — | — | — |
| 同上，K=4 + 伪决策（最终默认） | 0.97 | 0.98 | 1.00 | 0.99 | 0.78（easy 0.97 / medium 1.00 / **hard 0.14**） | 0.99 |
| K=4 + 伪决策，8 / 10 频带 | — | — | — | — | 0.30 / 0.70（更差、不稳定） | — |

VideoRepick 困难档（15 个方块、间距约 4 cm）读出头学不会，而 oracle 规则在真实未来里能 100% 找到答案 → 这是读出头的空间分辨率问题，与世界模型无关。
按判读规则 1 把这一层从主分析剔除（`experiment.yaml → stats.exclude: {VideoRepick: [hard]}`，在看到任何世界模型结果之前确定），附表仍按难度分层报告。

---

## 6. wandb

四个训练全部接入 wandb，配置集中在 `configs/experiment.yaml → wandb`（`project: owm_memory`，`mode: online`）：

| job_type | group | 记录内容 |
|---|---|---|
| `videosaur` | `videosaur_robomme` | 官方 Lightning logger（loss_featrec / loss_timesim / 验证损失），TensorBoard 和 CSV 仍然照常写 |
| `cjepa_predictor` | `cjepa/<tag>` | `train_loss`、`train_future_mse`、`train_masked_history_mse`、`val_future_mse`、每 epoch 耗时 |
| `lpwm` | `lpwm` | 仓库自己那套每 epoch 指标（各项 KL、`on_l1`、PSNR、LPIPS）+ `val/loss` |
| `readout` | `<任务>/<条件>` | 每 100 步的 train/val loss，结束时把 `test/acc_main` 等写进 summary，3 个种子同组便于对比 |

实现方式：`owm/wandb_utils.py` 统一初始化；LPWM 那边通过包住仓库自己的 `format_epoch_summary` / `log_line` 取数（**没有改 LPWM 仓库**）；VideoSAUR 由 `scripts/p5b_train_videosaur.sh` 把 `experiment.yaml` 里的 wandb 设置透传成命令行覆盖。wandb 挂了或没装都只打印一行提示，不会中断训练。

```bash
wandb login                      # 首次需要（~/.netrc 里已有凭据则跳过）
OWM_NO_WANDB=1 <任何命令>          # 临时关闭
WANDB_MODE=offline <任何命令>      # 断网时先存本地，之后 wandb sync
```
多卡 LPWM 只有 rank 0 建 run；`--probe`（显存探测）不建 run。

---

## 7. 小贴士
- 读出头可按任务并行：`CUDA_VISIBLE_DEVICES=k python scripts/p4_train_readout.py --conditions ... --tasks <Task>`（预测按任务/条件分文件保存，互不冲突）。6 任务 × 2 条件 × 3 种子在 5 张卡上约 40 min。
- 想完全回到规格原始设置做对照：`readout.fourier_freqs: 16`、`readout.resample_ids: false`、`readout.aug_offsets: []`、`temporal.future_offsets: [16, 32]`、`stats.exclude: {}`。
- 改了 `future_offsets` / `aug_offsets` / 决策规则后要重建索引：`python scripts/p2_gt.py --report-only`；改了 `future_offsets` 或 `context_frames` 后世界模型要重训、证据要重提。
- 不要在命令里用 `pkill -f <脚本名>` 之类的模式去杀后台任务——它也会匹配到正在启动的新命令自己。
- 大文件一律放 `~/data/owm_memory_data`（`/common/home` 配额 208 GB）。`cache/episodes` 约 160 GB，`data/generated` 约 44 GB。
