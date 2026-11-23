# Detection With Association Constraints · MDMT

当前完整数据中心监督实验：参见 [完整 MDMT 实验](FULL_MDMT_EXPERIMENT.md)，启动命令为 `bash tools/run_full_center_training.sh`。

`model/` 是作者更新提交
[`e2076af`](https://github.com/yyyyyxq/Instance-based-Association/tree/e2076afc7f59d9f82e17bc31cd8ddcf9a047fb5a/model)
中的源码为基础。当前仅修正了 `predict(class_threshold=...)` 放行背景类别的问题；
该接口修正不改变训练前向、损失或现有验证解码。训练直接调用作者的模型、`pad_targets`、Hungarian matcher 和 SetCriterion。
`training/` 负责 MDMT 数据加载、增强、训练接口和评估。旧版本的权重与新版架构不兼容。

## 数据与参数

从项目根目录运行命令。现有 `data/processed/MDMT/` 已包含三个训练所需清单。
若需要从原始 MDMT 重新准备，可执行 `python prepare.py --verify-images`。
使用官方测试 TXT 清单和现成的 ImageNet 权重时，训练可离线运行。

| 参数 | 当前主配置 |
| --- | --- |
| 模型 | ResNet50，作者更新后的 MVFF、GIQ、MVOQ |
| 优化器 | AdamW，学习率 1e-4，权重衰减 1e-5 |
| 训练 | 全局 batch 4，100 轮，翻转、裁剪和颜色增强 |
| 损失 | 作者实现，L1/GIoU 权重为 5/2 |
| MDMT 适配 | 两视角、类别无关、256 个查询，输入高 405 × 宽 720 |

训练、验证和测试有 12,026、1,930 和 5,883 组图像。完整验证集最多每组 221 个真实实例，
因此容量设为 256。论文报告的 25 查询不足以覆盖这份完整清单。单前景类别沿用现有 MDMT 清单，
避免跨视角类别冲突改变实例 ID。输入尺寸和小 batch 下冻结 ResNet 的 BN 运行统计是复现设置。
MVOQ 的实例条件查询与 MVFF 的 1×1 投影沿用作者更新代码中对公式的具体实现。

## 双 A6000 训练

在服务器的 CUDA PyTorch 环境安装依赖：`python -m pip install -r requirements.txt`。
如果使用了其他编号的空闲卡，调整 `CUDA_VISIBLE_DEVICES`。此前这台服务器的 P2P 通信曾卡住，
以下命令保留了已验证可用的 `NCCL_P2P_DISABLE=1`。训练入口使用 FP32。

先短测，一轮 16 组训练数据与 8 组验证数据，确认设备和通信：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --device cuda --batch-size 4 --epochs 1 --workers 0 \
  --limit-train 16 --limit-val 8 --output runs/author_gpu_check
```

长训前可用单张空闲 GPU 检查学习能力。下面固定一对真实训练图像，关闭增强，
使用当前完整模型和输入尺寸从 ImageNet 权重初始化，逐步报告框覆盖和正确关联：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/overfit.py --device cuda --steps 600
```

看 `raw_box_hits`（忽略分数的单视角框命中）、`raw_same_query_hits`（忽略分数的双视角同槽命中）、
`AP50` 和 `true_associations` 能否持续改善。此测试只衡量模型记住训练样本的能力。

固定多个场景可用 `--indices`，下面四组均不超过 25 个真实实例，索引对应完整训练清单。
它们仍是训练样本，测试结果不能作为验证集成绩或论文复现成绩：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/overfit.py \
  --config configs/mdmt_resnet50_no_dropout.json --device cuda \
  --indices 249 4140 7209 7955 --num-queries 25 --background-weight 0.1 \
  --steps 400 --report-every 100 --output runs/four_scene_check/report.json
```

监督、数据几何、解码和梯度的回归检查：`python -m unittest discover -s tests -v`。
检查通过只说明这些具体行为满足断言，不证明模型收敛或能复现论文成绩。

完整 MDMT 需要保留足够的查询。若稀疏样本上前景学习失败，可在同一固定图像对上测试
`--num-queries 256 --background-weight 0.1`。这一实验把匹配到填充背景的槽在分类和
可见性损失中的相对权重降到 0.1，按每组样本的有效权重归一化。真实实例在某视角不可见
时的负监督保持权重 1。Hungarian 匹配和框损失沿用作者实现。默认权重为 1，
直接调用原始损失；启用降权的训练应使用新输出目录。续训时读取该次运行保存的配置。

正式训练去掉样本限制，使用新目录。先跑 1～3 轮看完整验证集的 AP50、mAP、MDA 和正确关联数：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --device cuda --batch-size 4 --workers 4 --epochs 3 --output runs/mdmt_author
```

每轮写入 `history.jsonl` 和 `last.pt`。验证在 rank 0 跑完整验证集，
`best.pt` 按 MDA 选优，MDA 相同时看 AP50。若指标趋势正常，用同一 `torchrun` 启动行替换脚本参数：

```bash
train.py --device cuda --config runs/mdmt_author/config.json \
  --resume runs/mdmt_author/last.pt --epochs 100
```

这里 `--epochs` 是目标总轮数，且 `--resume` 默认沿用检查点所在目录。
`--batch-size` 是全局 batch，双卡计算时全局 4 对应每卡 2 组。

## 评估及推理

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py --device cuda \
  --checkpoint runs/mdmt_author/best.pt --split test \
  --output runs/mdmt_author/test_metrics.json \
  --export-predictions runs/mdmt_author/test_predictions

python tools/score_official_mda.py \
  --predictions runs/mdmt_author/test_predictions \
  --output runs/mdmt_author/test_official_mda.json
```

验证和测试输出 MDA、UCR、COCO mAP/50/75 及正确关联计数。官方旧 MDA 脚本单独记分。
推理使用 `python infer.py --checkpoint ... --images camera1.jpg camera2.jpg`，
共享 query ID 仅表示这些同步图像间的关联。
如需 ViT，使用 `--config configs/mdmt_vit_b16.json`，并在新目录运行。
