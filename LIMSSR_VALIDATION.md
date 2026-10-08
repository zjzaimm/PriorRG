# 集成验证记录

日期：2026-10-08，Asia/Shanghai。

环境位于 `D:\codex-envs\priorrg-limssr`，Windows / Python 3.12 / CPU。PyTorch 2.6.0+cpu、Transformers 4.43.3、PEFT 0.13.2、Lightning 2.3.0。确切核心依赖版本见 `requirements-limssr-validated.txt`。

## 已通过

- `pip check`：No broken requirements found。
- `main_github.py --help`、`main_single_sample_github.py --help`：退出码 0，新 LIMSSR 参数可用。
- Python compileall：退出码 0。原有 f1chexbert.py 中存在无效转义 SyntaxWarning，不阻止编译。
- 独立 LIMSSR 模块测试：8/8 通过。
- `smoke_test_limssr.py`：6/6 通过，退出码 0，结果在 `limssr_smoke_result.json`。

集成测试执行了真实模型构造和 forward：DINOv2 图像编码器、BERT 文本编码器、原 TemporalFusion、原 Perceiver、GPT2 EncoderDecoder、Qwen2Model、PEFT LoRA 和全部新模块。使用本地小型随机权重与合成图像/报告，只有依赖外部权重的评估器初始化被跳过。

具体证据：

1. 混合缺失批次的前向损失有限，本次值为约 3.1417（只用于运行检查，不是性能指标）。解码器收到 `[2,16,32]` memory，对应原 12 个 token + 4 个新融合 token。
2. 两个样本的输入可用性正确恢复为 `[1,1,1]` 和 `[1,0,0]`。
3. 缺失 token、融合 token、角色权重、投影和 MDA 混合参数均有有限非零梯度；LoRA B 参数收到梯度；optimizer 更新了新 memory_projection 权重。
4. 新参数全部在 optimizer 中，LIMSSR 参数独立学习率为 2e-4；没有闲置的数值评分头。
5. 历史影像/临床上下文的四种有无组合均能执行 beam generation；每次返回两条文本。
6. checkpoint 保存后能够 strict=True 加载到单样本推理模型，并执行生成。
7. 在新桥接路径中，把缺失历史影像、缺失上下文和文本 padding 的特征改为 NaN，不改变融合结果。
8. 原 FinetuneDataset/FinetuneCollateFn 读取本地合成数据，Lightning Trainer.fit 在 CPU 上完成一个真实训练 step，global_step=1。
9. 关闭 --use_limssr 时 baseline 可前向，不创建新模块参数。

## 尚未验证

没有运行完整规模 Qwen2.5-0.5B 预训练权重，没有加载正式 RAD-DINO/CXR-BERT/PriorRG checkpoint，没有训练完整 MIMIC 数据集，没有运行 CheXbert/RadGraph/COCO 临床评估栈，也没有验证 GPU、DDP、量化和正式指标。环境为 CPU 版 PyTorch；CUDA 训练需要安装合适的 CUDA 构建。

当前项目目录只有 demo 胸片，没有正式数据标注和完整权重，因此不能据此宣称已经可以直接复现正式实验或迁移后获得性能提升。新模块需要训练后才能正式推理；基于随机小模型的生成文本没有临床意义。
