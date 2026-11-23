# 全量训练非有限梯度：解释与恢复

## 日志能确认什么

用户提供的运行在epoch13、batch1058执行clip_grad_norm_时，检测到总梯度范数非有限而终止。不是CUDA OOM；OMP提示不是故障。terminate消息伴随异常退出，核心Python调用栈指向梯度检查，不支持将DataLoader视为本次主故障。

第8/9轮训练损失3.657→4.052→4.926且验证关联178→0，说明在崩溃前已经存在明显训练退化。仅凭这段日志不能断言具体层、算子、坏样本或唯一根因，也不能将其等同于模型参数已全变NaN。

两种情况都会触发旧报错：

1. 梯度张量本身含NaN/Inf。
2. 梯度元素仍有限，但计算FP32的L2范数发生溢出。

旧配置clip_grad=null，实际max_norm=inf，因此只作非有限检查，没有真正限制大梯度。短样例成功不保证长程优化稳定；此前全量配置的数值保护不充分。不能将error_if_nonfinite设为false来绕过，尤其inf/inf系数会进一步损坏梯度。

第13轮失败的step没有执行optimizer.step，且每轮原子保存，因此此前的last.pt（应是第12轮）仍可使用。根据日志，原best.pt应在第7轮；恢复脚本会检查并打印实际epoch。

## 本次补充

- training/gradients.py：正常批次仍用PyTorch快速裁剪；出错时区分真正NaN/Inf与范数溢出。只有在所有梯度元素有限且有有限裁剪阈值时，才用FP64计算范数和缩放，防止把有限大梯度错误清零。真正坏梯度仍会停止，绝不跳过或继续写入参数。
- 新启动的完整配置默认clip_grad=1.0；旧运行保存的配置不会被静默修改。
- train.py记录每轮rank0_grad_norm_max（裁剪前）与范数溢出恢复次数。
- 出错时自动记录rank、epoch、batch、阶段、样本、损失及逐参数梯度诊断JSON。
- --save-failure-batch额外原子保存模型与输入batch，供单batch反向复查；只在出错时生成较大的文件。
- --allow-optimization-change只允许续训时改变LR/clip，并要求新输出目录；修改来源与数值写入optimization_changes.json及检查点。正常同设置续训仍维持原规则。

这是一项数值稳定性补充，不是宣称已经定位并修复原始非有限梯度的具体算子。

## 用户运行步骤

通过Git同步全部修改，特别是training/gradients.py，服务器git pull后在项目根目录执行：

```bash
bash tools/recover_full_center_training.sh
```

默认从原runs/mdmt_resnet50_center_full/best.pt恢复，检查参数和优化器状态有限；学习率改为5e-5、裁剪阈值1.0。模型、数据、query、损失均不改，保留优化器矩估计并正确覆盖其LR。从best实际轮数继续到总计15轮。新结果保存在：

```text
runs/mdmt_resnet50_center_full_stable/
```

旧实验、旧best和last不覆盖。原best若是第7轮，则训练新阶段8–15；新目录best.pt是新阶段最佳，原第7轮基线仍保存在旧目录，应一起比较。

如果新目录已有last.pt，脚本自动从它继续。继续到总计30轮用：

```bash
EPOCHS=30 bash tools/recover_full_center_training.sh
```

SOURCE_CHECKPOINT可指定其它来源，例如原last.pt；默认选择best是因为第8轮开始已有退化。不要把本稳定性恢复分支与此前不裁剪/原LR实验混成一条未变更设置的结果。

## 再次出错时

恢复命令已经启用现场保存，新目录将出现：

```text
failure_rank0_epoch*_batch*.json
failure_rank0_epoch*_batch*.pt
failure_rank1_epoch*_batch*.json
failure_rank1_epoch*_batch*.pt
```

JSON先用于区分异常类型与参数。完整.pt仅在写入完成后才出现；进程被另一rank异常中断时，某rank可能只留下.tmp。复制已完成的snapshot到本地，可执行：

```bash
python tools/replay_failure_batch.py --snapshot <完整snapshot路径> --device cpu
```

工具用异常追踪重做本地batch前向/反向，不更新参数。它不回放DDP梯度归约，也不保证CPU能复现CUDA特有问题；必要时在原GPU上重放。

## 验证

28项回归测试通过，包含普通裁剪、真实NaN/Inf拒绝、有限FP32范数溢出恢复、无裁剪旧行为。

本地256query真实密集小流程完成旧检查点恢复、LR/clip变更、优化器LR核对、来源记录及训练保存。人工注入NaN梯度后，保护正确触发、JSON和snapshot正确保存；该合成错误来自测试hook，hook不保存到snapshot，因此其回放用于验证加载和计算流程，不是原始GPU失败复现。

本地没有此次全量训练的best/last，也没有A6000，尚未独立复现服务器第13轮异常；具体首个异常算子仍待真实现场或完整检查点分析。
