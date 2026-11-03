# 猫狗图像分类作业

项目只保留三种最终模型：DNN、CNN、RNN。网络结构、参数和验证结果见 [实验报告_三模型最终版.md](实验报告_三模型最终版.md)。

## 文件

- `src/catdog/models/`：DNN、CNN、RNN 的最终网络结构。
- `src/catdog/train.py`：训练入口。
- `src/catdog/evaluate.py`：验证集评估入口。
- `src/catdog/predict.py`：单图预测入口。
- `outputs_best/dnn/best.pt`、`outputs_best/cnn/best.pt`、`outputs_best/rnn/best.pt`：历史归档权重；本次正则化结果见 outputs_regularized 和实验报告。



训练全部三个最终模型：

```bat
python train.py --output-dir ..\..\outputs_regularized
```

只训练其中一个，例如 CNN：

```bat
python train.py --model cnn --output-dir ..\..\outputs_regularized
```

训练默认使用 64×64 图片、30 轮、batch size 32、AdamW、学习率 0.0003、随机种子 43，训练增强为水平翻转。当前代码采用以下正则化配置（本次 30 轮实验已完成）：

| 模型 | 权重衰减 | Dropout |
| --- | --- | --- |
| DNN | 0.001 | 混合模块 0.3，分类头 0.4 |
| CNN | 0.0005 | 分类头 0.35 |
| RNN | 0.001 | 分类头 0.4 |

仍按验证准确率选择 `best.pt`，准确率相同时选验证损失较低者；没有早停。上述命令将新结果写到 `outputs_regularized/<模型名>/`，保留原 `outputs` 作对照。不指定 `--output-dir` 时仍写入 `outputs/<模型名>/`，会覆盖该目录中的同名文件。

终端每轮打印训练与验证损失。每个模型的输出目录会保存：

- `best.pt`：验证准确率最高时的权重；`last.pt`：最后一轮权重。
- `history.jsonl`：逐轮的 `train_loss`、`val_loss` 和准确率，可直接读取绘图。
- `summary.json`：最佳准确率轮次、最佳验证指标、最低验证损失及其轮次、实际权重衰减和各层 Dropout。
- `training.log`：终端逐轮输出；`loss_curve.png`：训练/验证损失曲线，标出最低验证损失和最佳验证准确率轮次。

再次训练同一输出目录会覆盖其中的文件。原 `outputs` 的普通验证准确率为 DNN 69.4%、CNN 83.0%、RNN 70.0%；正则化后普通验证准确率为 DNN 69.2%、CNN 83.6%、RNN 70.4%。DNN 新配置未超过原配置；CNN、RNN 在本次验证中小幅提高。详细分析见实验报告。CNN 的翻转推理结果应另外比较。

验证和预测：

```bat
python evaluate.py --checkpoint ..\..\outputs_best\dnn\best.pt
python evaluate.py --checkpoint ..\..\outputs_best\cnn\best.pt
python evaluate.py --checkpoint ..\..\outputs_best\rnn\best.pt
python predict.py --checkpoint ..\..\outputs_best\cnn\best.pt --image ..\..\data\val\cat.1000.jpg
```

CNN 评估和预测自动使用原图与水平翻转图的平均输出；DNN、RNN 使用原图。上述 outputs_best 历史归档权重的验证准确率分别为 DNN 69.0%、CNN 86.2%、RNN 70.0%。

