# 作者公开模型 + MDMT训练外围

`model/`完整保留作者仓库固定提交[e2076af](https://github.com/yyyyyxq/Instance-based-Association/tree/e2076afc7f59d9f82e17bc31cd8ddcf9a047fb5a/model)的12个Python文件。没有本地新增的PE、实例残差、参考中心、中心监督、背景降权或logits损失。`AUTHOR_SOURCE.json`记录逐文件来源和SHA256，`tools/check.py`检查源文件是否改变。

这份公开源码本身与论文文字有若干差异，本次按要求保留源码，不再自行改写模型或损失。因此这是“作者公开版本基线”，不能宣称已经完成论文指标复现。

## 目录

```text
model/             作者固定版本的模型、匹配、损失、GT填充
training/          数据加载、训练/评估封装、DDP、梯度检查
configs/           ResNet50、ViT两份完整MDMT配置
train.py           训练、保存、恢复
evaluate.py        验证/测试、预测导出
infer.py           同步图像组推理
prepare.py         原始数据生成清单
tools/            check.py、prepare_data.py、score_official_mda.py
tests/            作者模型与训练外围的必要回归测试
data/             原始数据和已准备清单（保留）
weights/          ImageNet预训练权重（保留）
runs/             本基线的新实验输出
改正/             改进版代码备份、30轮结果、审计及备份校验
```

## 配置边界

- 两视角、单前景类别，完整train/val/test分别12026/1930/5883组，不截断GT。
- 作者模型默认25个query保持不改。MDMT配置使用256，因为完整验证集最大221个联合实例；这是显式数据容量适配，仍需作者确认论文MDMT实际query设置。
- ResNet输入405×720；ViT输入416×720，避免patch16丢边界。
- 作者原损失：所有槽等权分类和可见性BCE；L1/GIoU默认5/2。原有概率clamp也保留，没有暗中替换数学/数值实现。
- AdamW：LR1e-4、weight decay1e-5、目标100轮、全局batch4，双卡每卡2；FP32。
- 开启同步裁剪/翻转及颜色增强。外围默认冻结BN运行统计、clip_grad=1.0；这两项是显式工程设置，论文没有给出对应细节。
- DDP仅对回归项按全局可见框数缩放，不增加损失项。验证在rank0执行，best按MDA优先、AP50次优。
- 常用推理/评估使用外围decode；作者`predict`函数也原样保留。
- MDA使用逐视角一对一匹配，空GT/空预测关联帧排除并记录；COCO AP使用maxDets=100。官方legacy分数另算，不保证二者完全相同。

## 先检查，再做双卡小训练

在服务器项目根目录、已安装匹配的CUDA torch/torchvision环境中：

```bash
python -m pip install -r requirements.txt
python tools/check.py
python -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --config configs/mdmt_resnet50.json --device cuda \
  --batch-size 4 --workers 0 --epochs 1 \
  --limit-train 16 --limit-val 8 --progress bar \
  --output runs/author_smoke
```

可选单卡真实最密集样本前后向：`CUDA_VISIBLE_DEVICES=1 python tools/check.py --device cuda --backward`。ViT使用`--config configs/mdmt_vit_b16.json`。

## 新的完整基线训练

小训练成功后，从ImageNet重新初始化，使用新目录；不要接着小训练或30轮改进版权重训练：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --config configs/mdmt_resnet50.json --device cuda \
  --batch-size 4 --workers 4 --epochs 100 --progress bar \
  --save-failure-batch --output runs/mdmt_resnet50_author
```

100为论文目标预算，不意味着建议不检查指标就等待100轮。每轮查看完整验证结果，保留日志。源码运行成功不保证达到论文成绩。

同一基线断点续训：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --config runs/mdmt_resnet50_author/config.json \
  --resume runs/mdmt_resnet50_author/last.pt \
  --device cuda --epochs 100 --progress bar
```

## 评估与原始数据准备

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --checkpoint runs/mdmt_resnet50_author/best.pt --split val --device cuda \
  --output runs/mdmt_resnet50_author/val.json

# 最终评估再使用--split test；可加--export-predictions导出官方适配器输入。
# 数据已准备时不用重跑；从原始数据重建清单：
python prepare.py --verify-images
```

## 备份与同步

改进版代码位于`改正/改进版代码/`，完整30轮模型/日志及诊断位于`改正/30轮实验结果/`。它们与当前作者基线隔离；当前训练入口会拒绝带改进模块/损失的配置和检查点。

`改正/`、`data/`、权重和新`runs/`被Git忽略。本次代码同步需要包括新增文件和删除记录，使用`git add -A`检查后提交。服务器`git pull`不会自动移动或删除其被忽略的旧实验；本次整理作用于本地。备份说明见`改正/说明.md`。
