# 作者公开源码：MDMT车辆检测与跨视角关联试跑

当前方案按用户向一作确认的信息设置：**25个query，只检测车，其余为背景，分类头共2个输出。** 暂时使用手头这版MDMT；先保留双视角联合车辆数不超过25的图像组，组内车辆GT全部保留。不是作者早期数据版本，也不是完整新版MDMT评测。

`model/`的12个Python文件保持作者固定提交[e2076af](https://github.com/yyyyyxq/Instance-based-Association/tree/e2076afc7f59d9f82e17bc31cd8ddcf9a047fb5a/model)不变。`AUTHOR_SOURCE.json`记录来源与校验值，`tools/check.py`检查是否改变。模型没有本地PE、实例残差、中心监督、背景降权或logits损失。

## 数据与类别

- 只将原始XML的`car`标注作为真实目标（类别0）；person、bicycle、bus不提供目标框监督，作为非目标背景。无目标填充类别为1，不存在额外“非车目标类”。
- 原始跨视角ID保持不变。同ID只有标为car的视角才提供框与可见性正监督；不会将另一视角的非car标注重写为car。
- 先筛选每个视角的car，再取跨视角ID并集计算M。M<=25保留整组，M>25排除整组；不截断组内车辆GT，不拆分实例，不打乱原train/val/test划分。
- 已标注但没有car的图像组可作为负样本保留。本次实际保留清单中没有这类空组。
- 使用XML标注测试集，不能拿只有车辆的预测对全类别官方TXT打分。预测导出携带类别元数据，官方适配器会拒绝该错误组合。
- 旧类别冲突审计仍保留在旧目录；本次明确的car-only协议无需全类别统一标签，因此不再被四类一致性预检拦截。但原始ID映射与当前数据版本是否和作者早期数据一致仍待确认。

清单位置：`data/processed/MDMT_car_q25/`。准备脚本默认生成此协议，旧数据和清单不覆盖。

|划分|原始图像组|保留图像组|保留序列数|最大联合car实例|
|---|---:|---:|---:|---:|
|train|12026|3066|8|25|
|val|1930|95|1|25|
|test（XML）|5883|1643|5|25|

验证仅保留序列22的95组。因此本轮只作为当前数据的25-query车辆子集试跑，不与论文完整MDMT指标或之前全目标/256-query结果直接横比。具体筛选数与原标注哈希在`audit.json`。

## 目录

```text
model/          作者模型、匹配、损失、GT填充（原样）
training/       数据加载、DDP、训练/验证封装、梯度检查、指标
configs/        ResNet50、ViT的car-only/Q25配置
train.py        训练与恢复
evaluate.py     验证/测试、导出
infer.py        同步图片组预测与画框
prepare.py      准备车辆子集（默认）
tools/          check.py、prepare_data.py、score_official_mda.py
tests/          数据、原损失、梯度、保存恢复等回归检查
data/ weights/  当前数据与ImageNet权重（保留）
runs/           新实验输出
改正/           旧改进代码、30轮结果及旧类别配置备份
```

## 服务器准备与训练

Git同步本次全部改动后，在服务器项目根目录执行。模型权重、原始数据需要已存在；新生成清单不通过Git上传，由服务器本地生成。

```bash
python prepare.py
python tools/check.py
python -m unittest discover -s tests -v
```

先跑15轮看趋势，从ImageNet重新初始化，不接旧实验：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --config configs/mdmt_resnet50.json --device cuda \
  --batch-size 4 --workers 4 --epochs 15 --progress bar \
  --save-failure-batch --output runs/mdmt_car_q25_author
```

配置目标默认100轮，命令的15轮是初步预算。保持作者原损失、dropout0.1、AdamW LR1e-4/weight decay1e-5；开启裁剪/翻转/颜色增强。全局batch4、双卡每卡2、FP32。外围冻结BN运行统计并裁剪梯度范数1.0，仍是需作者确认的工程设置。ResNet输入405×720，ViT输入416×720。

每轮写入`history.jsonl`、`last.pt`，MDA优先/AP50次优保存`best.pt`。继续同一实验到总计100轮：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29643 train.py \
  --config runs/mdmt_car_q25_author/config.json \
  --resume runs/mdmt_car_q25_author/last.pt \
  --device cuda --workers 4 --epochs 100 --progress bar
```

单独验证：

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --checkpoint runs/mdmt_car_q25_author/best.pt \
  --split val --device cuda --output runs/mdmt_car_q25_author/val.json
```

最终测试将split改为test，读取相同car-only/Q25筛选后的XML测试清单。模型常用推理由外围decode完成，未改作者predict接口。

## 复现边界与备份

当前二分类是“只有car为目标”，与此前“把所有真实物体都合并成前景”的设置不同。类别数同为1也不代表监督相同，旧权重/旧清单不能接着本实验训练。

保留作者损失意味着概率clamp的原实现也保留，本次不混入此前数值改进。公开源码与论文文字的既有差异仍存在，不等于已确认论文成绩可复现。

`改正/`、数据、权重和runs被Git忽略；使用`git add -A`检查代码变更与删除记录后同步。原30轮结果位于`改正/30轮实验结果/`，旧改进代码位于`改正/改进版代码/`。源码校验、模型运行与测试通过只证明相应计算行为，不保证训练效果。
