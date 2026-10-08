# 验证记录

日期：2026-10-08（Asia/Shanghai）。

环境：Windows，Python 3.12，PyTorch 2.4.1+cpu。PyTorch 安装在工作目录 `.validation_deps`，未修改系统 Python。安装环境只用于本地验证，不包含在交付压缩包中。

结果：Python compileall 成功；8 项 unittest 全部通过，退出码 0。

1. 三模态所有七种非空组合的有限输出与融合 token 形状。
2. 任意模态名称、两模态、三维输出、独立 MDA 调用。
3. 梯度到达缺失 embedding、融合 embedding、已观测投影、backbone、角色权重和混合参数；缺失投影没有梯度。
4. 非法 mask、被标记为可用但为 None 的模态、目标 shape 广播错误被拒绝。
5. 默认 triplet 与直接循环参照计算一致；论文公式在 K=2 的恒定性质被核对。
6. 混合掩码批次还原后与逐样本前向一致；缺失行填入 NaN 不影响输出。
7. 观测块先于缺失块；完全缺失的模态可输入 None。
8. B=1 或 K=1 时 triplet 返回可反传的零，梯度有限。

测试使用 ToyTokenizer 和一个前缀累积的 ToyBackbone，不下载真实 LLM。没有验证预训练 Qwen 权重、transformers/peft 版本组合、GPU、分布式训练、数据集评分或迁移效果；没有声称复现论文性能。完整方法仍然依赖预训练语言模型，不能用测试中的 ToyBackbone 替代并保留同样的方法主张。

在装有 PyTorch 的环境中重新运行：

```powershell
python -m unittest limssr_transfer.test_modules -v
```

本工作目录的隔离依赖验证命令（Python 可执行文件路径见桌面运行时配置）：

```python
import sys
sys.path.insert(0, '.validation_deps')
import unittest
suite = unittest.defaultTestLoader.loadTestsFromName('limssr_transfer.test_modules')
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
```
