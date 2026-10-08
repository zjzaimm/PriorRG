# PriorRG + LIMSSR 接入说明

本次接入范围为 PriorRG 的第二阶段报告生成，以及对应的单样本推理。第一阶段对比预训练保持原样。用 `--use_limssr` 启用；不传该参数则不创建新模块，也不加载额外 LLM。

## 已接入的方法

1. 在 PriorRG 提取当前影像、历史影像和文本特征后，单独保留当前/历史影像特征，并用 `pri_idx` 还原每个样本真实的历史影像可用性。
2. 当前影像、历史影像、临床上下文作为三个输入组。临床上下文使用原文本编码器 attention mask 去掉 padding 后压缩；去掉 `[INDICATION]`、`[HISTORY]`、`[SEP]` 等结构标记后为空的上下文，才标记为缺失。`no acute disease` 等有内容的文本仍算观测信息。
3. PCMI 按“提示 → 已观测块 → 缺失块 → 融合 slots”构造输入，经 Qwen/LoRA 推断潜在表示；不向该路径输入目标报告。
4. LMRF 保留 K 个融合 token 和可学习角色权重。MDA 保留掩码残差校正、跨模态注意力、模态门控和缺失置信度。
5. 将校正后的主路 slots 与辅助特征按可学习 sigmoid 混合系数融合，再投影回 PriorRG hidden_size，追加到原有粗到细 Perceiver 的输出中，让原 GPT2 报告解码器读取。原输出长度 `3*num_latents` 变为 `3*num_latents + limssr_fusion_tokens`。
6. 训练损失为 `原报告生成 CE + consistency_weight*特征一致性 + metric_weight*token triplet`。特征一致性用两个分支归一化表征的 MSE，是针对报告生成的适配；不是原论文的数值评分一致性。

原 LIMSSR 的两个标量评分头已在 `feature_only` 模式下移除。两条路径都参与报告生成梯度；本次没有声称移植了原数值评分任务，也没有使用两次解码的 token KL 一致性。

## 文件位置

- `models/limssr/modules.py`：可移植核心模块，新增 feature_only 模式。
- `models/limssr_bridge.py`：PriorRG 特征压缩、可用性判定、解码 memory 桥接和辅助损失。
- `models/model_github.py`：报告生成/单样本 forward 的接入点；新参数使用独立学习率组。
- `tools/utils_github.py`、`main_single_sample_github.py`：新 CLI 参数。
- `main_github.py`：无 CUDA 时使用 CPU Trainer。
- `smoke_test_limssr.py`：使用真实小型架构的离线训练与生成验证。

## 重要适配差异

- 保留 Transformers 4.43.3，因此默认推理骨干为 `Qwen/Qwen2.5-0.5B`（Qwen2 架构）。原论文默认 Qwen3-0.6B 需要更新 Transformers，尚未验证更新后与 PriorRG 自定义 BERT 的兼容性。
- 图像 patch 和文本 token 数不同，因此各自自适应平均压缩到 `limssr_tokens=16`；这是空间/文本序列压缩，不是原 AQA 的时间片抽样。
- 两个已有临床文本字段合并为一个 context 输入组。只有一个字段缺失时仍存在临床上下文；本版没有拆成 indication/history 两个独立缺失组。
- 主路与辅助路在特征空间融合，并以特征一致性适配语言生成。混合系数约束为 [0,1]。
- 投影使用 LayerNorm，以支持不同缺失组合的小分组；边界与缺失 token 为独立 embedding 参数。详见 `models/limssr/README.md` 的论文/代码对照。
- 真实可用性只来自输入，未加入随机模态丢弃；不需要完整模态 teacher 或重建目标。
- 历史影像编码若原项目因损坏的索引/不存在的文件失败，应修复输入数据；本版不会把 I/O 错误默认为缺失。
- 原 PriorRG 的 Perceiver 仍采用其原有 context 处理；新桥接路径排除文本 padding，但没有更改原 Perceiver 的 padding 策略。

## D 盘环境与离线运行

本次创建的环境：`D:\codex-envs\priorrg-limssr`。pip 缓存：`D:\codex-cache\pip`；临时下载：`D:\codex-cache\tmp`；小型测试资产：`D:\codex-cache\priorrg-smoke`。

在 PowerShell 中执行：

```powershell
Set-Location 'C:\Users\44558\Documents\ChatGPT\paper\PriorRG-main'
& 'D:\codex-envs\priorrg-limssr\Scripts\python.exe' smoke_test_limssr.py
& 'D:\codex-envs\priorrg-limssr\Scripts\python.exe' -m unittest models.limssr.test_modules -v
```

smoke 使用本地随机初始化的 DINOv2/BERT/GPT2/Qwen2 和真实 LoRA。仅跳过需要外部权重的临床评估器初始化；图像编码、文本编码、Perceiver、GPT2、LIMSSR、优化器、数据加载和 Lightning 训练均执行真实代码。生成文本不具备临床意义。结果写入 `limssr_smoke_result.json`，环境与结果说明见 `LIMSSR_VALIDATION.md`。

## 正式训练

当前目录只有两张 demo 胸片，没有完整 MIMIC-CXR 标注、影像目录、RAD-DINO/CXR-BERT/PriorRG checkpoint，以及 CheXbert/RadGraph 评估权重。正式训练需要配置 README 中的数据和权重，并安装原 `requirements.txt` 中的临床评估依赖。D 盘环境目前为核心模型运行/离线验证环境，不包括完整临床评估栈。

设置所有下载位置在 D 盘：

```powershell
$env:HF_HOME='D:\codex-cache\huggingface'
$env:PIP_CACHE_DIR='D:\codex-cache\pip'
$env:TEMP='D:\codex-cache\tmp'
$env:TMP=$env:TEMP
```

将原来的报告生成训练命令补上下面这些参数即可启用接入：

```text
--task report-generation-gpt2 --phase finetune --use_limssr
--limssr_model_path Qwen/Qwen2.5-0.5B
--limssr_tokens 16 --limssr_fusion_tokens 4 --limssr_lora_rank 16
--limssr_lr 2e-4 --limssr_consistency_weight 0.1 --limssr_metric_weight 0.01
```

`limssr_model_path` 独立于原项目 `ckpt_zoo_dir`，可以填 HuggingFace ID 或绝对本地模型目录。建议将本地模型、`project_name` 输出目录及 checkpoint 目录均配置到 D 盘。

`--load` 可以使用原 PriorRG 权重初始化已有层，新增桥接模块需要训练；正式推理必须使用启用 LIMSSR 后训练得到的新 checkpoint。单样本入口会拒绝缺少新模块参数的 checkpoint，防止静默使用随机桥接权重。训练/单样本推理的骨干、tokens、LoRA rank 等结构参数必须一致。旧 checkpoint 用于 baseline 推理时不要传 `--use_limssr`。

## 复现边界

跑通烟雾测试只证明程序链路可执行，不证明新方法优于 PriorRG 或已经复现 LIMSSR 指标。正式实验应保留 baseline，对照辅助损失权重及缺失组合分别做消融，并单独记录临床质量指标。原源码在 generation setup 中用 test split 作为 val_set；正式实验前应检查并纠正数据划分，避免用最终测试集选模型。
