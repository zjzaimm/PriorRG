# LIMSSR 可移植模块

**PriorRG 接入版本：** 本目录随 PriorRG 使用 Transformers 4.43.3，默认骨干改为 Qwen2.5-0.5B；新增 `feature_only` 模式。实际报告生成接法见项目根目录 `LIMSSR_INTEGRATION.md`。以下为独立模块的设计说明，其中 Qwen3 示例需要更新 Transformers，不能直接在当前验证环境中运行。

这是基于论文方法和官方代码接口重新实现的 PyTorch 模块包，用于后续接入另一个项目。它不是原仓库的逐行复制，也不能直接加载原作者的完整 checkpoint。当前尚未接入目标项目，尚未验证迁移后的任务精度。

来源：

- 论文 v1：https://arxiv.org/html/2605.00434v1 ，重点为 §3.4–3.7。
- 官方代码：https://github.com/XuHuangbiao/LIMSSR 。核对日期：2026-10-08；代码来源为当时读取的 main 分支，未固定 commit。
- 核对文件：models/llm_aqa_model.py、models/loss.py、models/triplet_loss.py、train_llm.py、main_llm.py。
- 使用或发表时保留 LIMSSR 引用，并将本包的移植改动与原论文方法区分。

## 创新点与代码对应

| 方法 | 接口 | 论文位置 | 依赖关系 |
|---|---|---|---|
| PCMI：提示引导的缺失模态潜在补全 | `PromptGuidedModalityImputation` | §3.4，公式 5–7 | 需要 tokenizer、预训练语言模型；构造输入后必须执行 backbone，才产生推断表征 |
| LMRF：多融合 token 与角色加权 | `MultidimensionalRepresentationFusion` | §3.5，公式 8 | 输入为 backbone 末尾 K 个融合位置的隐藏状态 |
| MDA：掩码条件残差、跨模态注意力与置信度、双路径评分 | `MaskAwareDualPathAggregation` | §3.6，公式 9–14 | 可独立接收主表征和各模态表征；自身不完成缺失模态补全 |
| 双路径一致性、token 度量正则 | `limssr_loss` / `token_metric_loss` | §3.7，公式 16–17 | 回归损失可替换；token 正则保留论文/代码两个版本 |
| 完整方法连接器 | `LIMSSRAdapter` | §3.4–3.7 | 自动组合以上模块，按不同样本 mask 分组执行并还原顺序 |

论文的“不依赖完整模态监督”也是训练设定贡献，不能只靠一个模块实现。数据加载器必须允许真实缺失模态，mask 必须来自真实可用性，不能把实际不存在的模态通过随机操作标记为存在。本包不需要缺失模态的重建目标，也不引入完整模态 teacher。

## 输入输出合同

`modality_dims` 是有序字典，如 `{'visual': 768, 'audio': 768, 'flow': 1024}`；字典顺序就是 mask 列和输出 modal_features 的顺序。

- `features`：相同键的字典，每个值是 `[B,T,C_m]` 浮点 tensor。整批都缺失的模态可为 `None`。同一批内部分缺失时，保留形状一致的 tensor，缺失行可以填零，其对应 mask 必须为 0。
- `mask`：`[M]`（整批共享）或 `[B,M]`（逐样本）。1 表示观测到，0 表示缺失。每个样本至少有一个观测模态。
- 模态数、输入维度可配置。所有模态必须提前对齐为同一个正整数 T；当前没有有效帧 padding mask 接口，不能把时间 padding 当作真实帧直接输入。
- 所有模块和输入置于同一设备。封装适用于单设备；没有验证 device_map 自动分片、分布式训练和量化。
- 返回 `prediction`、`prediction_main`、`prediction_aux`：`[B,out_dim]`；默认 `[B,1]`，不自动 squeeze。
- 返回 `main_feature`、`aux_feature`：`[B,D]`；`fusion_tokens`：`[B,K,D]`；`modal_features`：`[B,M,D]`。
- D 为 LLM 隐藏维度，K 默认为 4；注意力 head 数必须整除 D。
- backbone 必须支持输入 embedding，并返回末层隐藏状态；预训练因果语言模型是预期用途。离线 ToyBackbone 仅用于接口测试，不构成 LIMSSR 的 LLM 方法。

## 完整接入示例

仅基础模块依赖 `torch`；需要真实 LLM 时安装 `transformers` 和 `peft`，并使用支持 Qwen3 的版本。未提供锁定环境，也未验证真实 Qwen 加载；不要把接口测试视为论文复现成功。

```python
from limssr_transfer import LIMSSRAdapter, load_hf_backbone, limssr_loss

backbone, tokenizer = load_hf_backbone('Qwen/Qwen3-0.6B')
adapter = LIMSSRAdapter(
    {'visual': 768, 'audio': 768, 'flow': 1024},
    backbone, tokenizer,
    task_description='action quality assessment',
).to(device)

# 在创建 optimizer 前注册 adapter，使其参数被 optimizer 管理。
# encoder 输出对齐后的特征，使用真实可用性的 mask。
out = adapter(features, mask)
losses = limssr_loss(out, normalized_score)
losses['total'].backward()
```

可以运行 `python -m limssr_transfer.example --model <本地模型目录>` 查看一轮训练示例；不传目录则尝试下载 Qwen。示例只使用随机输入演示调用，不证明有效性。

## 保留另一个项目的预测头

```python
# __init__：注册 adapter 和原来的预测头。
# 若原预测头维度不是 D，先添加并注册 nn.Linear(D, 原维度)。
out = self.adapter(features, mask)
prediction = self.existing_head(out['main_feature'])
loss = self.original_task_loss(prediction, target)
```

这种接法保留 PCMI、LMRF 和 MDA 的主路修正，但如果只用 main_feature，辅助路径不参与最终预测，其独有参数也不会从主任务损失获得梯度。它应描述为“部分移植”，不能宣称已经使用完整 MDA。要保留双路径，用相同输出空间的主/辅预测头，并加入对应的一致性项和最终混合。

也可以单独调用 MDA：

```python
from limssr_transfer import MaskAwareDualPathAggregation

self.aggregation = MaskAwareDualPathAggregation(
    hidden_dim=256, num_modalities=2, bounded_scores=False,
)
# main: [B,256]，modal: [B,2,256]，mask: [B,2]
out = self.aggregation(main, modal, mask)
```

这只是借用掩码感知聚合。若没有 LLM 和补全隐藏状态，不包含 PCMI/LMRF 的核心推断机制。分类任务需要适配 logits 的最终融合和一致性目标，不能直接沿用默认 sigmoid 与 MSE。

## 与官方实现的明确差异

1. 模态数、模态名称、输入维度泛化；官方固定 RGB/audio/flow。新增逐样本 mask 支持，通过同 mask 分组，保证所有已观测块排在缺失块前，保留因果注意力的方向。分组会增加前向调用次数。
2. 边界、缺失、融合位置使用独立可训练 embedding 参数；官方把特殊 token 加入 tokenizer 的词表。机制保留，但初始化和参数位置改变，因此原 checkpoint 不直接兼容。
3. 默认保留两层 1×1 Conv + ReLU 投影，但使用 LayerNorm 支持小批量和分组。`projection_norm='batch'` 可采用官方 BatchNorm 形式；训练时每个分组的 B×T 必须大于 1，不同 mask 分组会改变 BN 统计量。
4. HF 加载器使用 `AutoModel` 而非 `AutoModelForCausalLM`，跳过无用的全词表 logits；LoRA 加在 q/k/v/o projection，输入 embedding 保持可训练。没有添加语言生成目标。
5. MDA 残差 gate 按官方代码使用单 Linear + Sigmoid；论文文字描述 gate 为两层 MLP。缺失置信度参数 0.6 的实际初值为 sigmoid(0.6)，约 0.646；保留此代码行为。
6. 默认 `constrain_mix=False`，沿用官方不受约束的全局混合参数，初值 0.7。即使两个分支 sigmoid，最终分数也不保证在 [0,1]。`constrain_mix=True` 是显式移植改动，将混合系数限制为 [0,1]。
7. 默认 `bounded_scores=True` 对应归一化分数回归；目标必须按任务分数范围归一化。`bounded_scores=False` 输出无界数值，适合另行设计的目标空间。

## 论文与代码的 token 正则差异

默认 `metric_mode='official_triplet'` 对照仓库 loss.py：同一个 slot 在不同样本间为正对，不同 slot 为负对，使用余弦距离的 hardest triplet。B<2 或 K<2 时没有有效 triplet，返回可反传的零；这是对官方小批量边界行为的修正。

`metric_mode='paper_eq17'` 按论文公式字面实现：对每个样本的每个 token，计算其他 token 相似度的 max − min + margin，再取正部并沿 K 求和、沿 B 求均值。该公式与官方跨样本 triplet 不等价。特别是正 margin 下其下界为 K×margin，K=2 时恒定，不能直接将它理解为可靠的防塌缩目标。这里保留原式供核对，不悄悄改成别的损失。

默认权重为任务 10、一致性 1、token 正则 1。分类或其他非回归任务保留原任务损失，自行适配分支一致性；token 正则可以单独调用。混合 mask 分组后，token 正则仍在还原顺序后的整批 tokens 上计算。

## 验证与后续迁移

离线测试命令：`python -m unittest limssr_transfer.test_modules -v`。测试覆盖 7 种非空组合、逐样本 mask 还原、缺失行 NaN 不泄漏、None 模态、关键参数梯度、损失参照计算、小批量边界、任意模态数和形状错误。

测试不下载 LLM、不依赖数据集，不验证真实 Qwen/LoRA 的加载兼容性、CUDA 显存或论文指标。当前环境与测试结果记录在 `VALIDATION.md`。

拿到下一个项目后，需要检查其 encoder 输出、时间对齐、任务目标、预测头维度、真实 mask 来源和训练配置，再确定插入哪个模块。建议对照 baseline、仅 MDA、PCMI+LMRF、完整方法、完整方法去掉正则分别做消融；通过独立验证集选模型，最终测试集只报告结果。
