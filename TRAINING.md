# 训练说明

本次补充的是独立训练入口 `train.py`。原有推理脚本保留；新训练流程不导入旧的
`utils` / `dataset`，因此不会在导入时强制切换到 CUDA。

## 1. 准备数据

现有数据格式可以直接读取，无需移动、复制或改名：

```text
boostnet_labeldata/
  data/training/*.jpg
  data/test/*.jpg
  labels/training/*.jpg.mat
  labels/training/filenames.csv
  labels/training/landmarks.csv
  labels/training/angles.csv
  labels/test/...
```

在本机设置路径后，可以先只检查数据和划分，不创建网络：

```powershell
$dataRoot = "D:/datasets/boostnet_labeldata"
python train.py --stage wsm --data-root "$dataRoot" --invalid-samples skip --check-data --output-dir saved_model/data-check
```

`--data-root` 接受 boostnet_labeldata 目录，也接受包含它的父目录。
默认只使用 `data/training` / `labels/training`，从中留出 20% 的文件名病例组作为验证，
`data/test` 下的原 128 张图像不参与模型选择。种子为 `--seed 2019`，所有训练阶段
使用相同的病例划分。先划分整张图像，再产生局部 patch，避免 patch 跨集合。
同一日期/编号的 A、B 等 AP 视图、大小写及 `(2)` 后缀归为同一组；coronal1～17
缺少可确认的病例编号，因此同一日期批次保守地放在同一组。这是文件名推断分组，
不等于验证了真实患者身份。若有患者 ID，应先按真实患者建立独立训练/验证目录。

若已有独立 `data/validation` / `labels/validation`，加 `--val-split validation`。
也保留旧式 `--train-dir` / `--val-dir`，既支持 `<split>/images` + `<split>/labels`，
也可分别直接指向原生的 `data/<split>`。两种输入方式不能混用。

程序会拒绝重名图像 ID、训练/验证 ID 重叠和内容完全相同的跨集合图像，但无法
自动识别同一患者的不同影像，患者级划分需要由数据提供者保证。

标注顺序：17 个椎体从上到下；每个椎体为左上、右上、左下、右下四点。
坐标为像素 `(x, y)`，不是 `(row, column)`。原加载器没有进行减一转换，因此新入口
默认零起始；若 `.mat` 实际保存 MATLAB 一起始坐标，应显式加 `--coordinate-base 1`。
`.mat` 读取变量 `p2`；`.npy` 不允许对象数组。默认 `--label-format auto` 优先读取
MAT/NPY，缺少像素标注时按 CSV 文件名配对。可用 `--label-format csv` 明确选择 CSV：
`filenames.csv` 一行一个图像文件名，`landmarks.csv` 一行 136 列，先 68 个归一化 x、
后 68 个归一化 y；乘原图宽、高还原像素坐标，不受 `--coordinate-base` 影响。
`angles.csv` 中的三个 Cobb 角不是当前热图网络训练目标，不会作为角点坐标读取。

### 异常标注策略

默认 `--invalid-samples error` 会在训练前停止，并写入 `data_audit.json`。
明确指定 `--invalid-samples skip` 才排除问题样本，报告保留路径和原因。
不会自动排序椎体、裁剪越界坐标、截断点数，也不会改写原始 MAT/CSV。

本机 MAT 与 CSV 检查一致：原训练 481 张中 24 份未通过（23 份顺序、1 份越界），
457 张可用；默认分为 374 张训练（183 组）、83 张验证（46 组）。原测试集 128 张中
124 份通过、4 份顺序异常；这些测试数据不参与默认训练/验证划分。
另一组 98 张测试图像的已解压标注为 76/84/56 点，当前 17 椎体、68 点训练入口
会拒绝这些点数，不能直接与标准训练集混合。

除合成回归测试外，已用上述真实训练数据验证六个阶段各一个训练、验证 batch，
采用 CPU、96×64 调试尺寸；没有进行完整训练，也未计算论文的最终 Cobb 角 SMAPE。

## 2. 训练顺序

在项目目录激活新环境后，使用 README 中的六条命令。A、B、C、D、E 独立优化。
例如开始 A 的正式训练（使用默认完整输入尺寸）：

```powershell
python train.py --stage wsm --data-root "$dataRoot" --invalid-samples skip --output-dir saved_model/wsm
```

Mask R-CNN 在 A 训练结束后训练；通过 `--wsm-checkpoint` 加载冻结的 A，训练和验证
都使用 A 的预测前景遮罩。这里不使用验证集人工标注生成输入遮罩。

局部 D/E 按真实四角点生成外接框，框的总宽、高各增加 `--crop-margin 150` 个
原图像素，再缩放到 255×255；这一尺寸来自原推理代码。训练时允许裁剪中心扰动，
验证时固定裁剪。两阶段局部网络各自训练，并非把 D/E 串接反向传播。

默认五个热图网络都用 U-Net。原 `main.py` 的 B 使用 ResUNet，如需直接配合旧脚本，
可在训练 B 时指定 `--arch resunet`。否则将旧脚本中的 B 实例改为 `UNet(1, 2)`。
可选 ResUNet 保留历史计算方式以兼容其权重；默认 U-Net 训练不依赖它。

训练只调整训练集上的参数；验证使用 `eval()` 和禁用梯度的前向传播。
学习率调度和最佳模型选择依据验证集，默认连续 40 轮未改善时早停。

## 3. 参数来源与可调设置

| 设置 | 默认值 / 来源 |
| --- | --- |
| 输出通道 | A/B/D 为 2，C 为 18，E 为 5；来自论文 |
| 网络输入尺寸 | 来自原 `main.py` 的阶段配置；参数顺序为高、宽 |
| U-Net 输出 | 保留原网络 softmax，背景在最后一通道 |
| WSM / TSM | 连续椎体轮廓包络，膨胀后高斯模糊；论文没有给出精确公式 |
| WSM 膨胀 / 模糊 | 中位椎体宽度的 0.75 / 0.20 倍；本次重建的可调默认值 |
| TSM 膨胀 / 模糊 | 中位椎体宽度的 0.25 / 0.08 倍；本次重建的可调默认值 |
| 局部点高斯 sigma | 输出空间 6.67 像素，三倍 sigma 截断；参考旧代码 40 像素核 |
| 热图损失 | `(BCE + Focal) / 2`，Focal alpha=0.25、gamma=2，目标 >0.01；参考 dental |
| 优化器 | SGD，lr=0.01、momentum=0.9、weight_decay=1e-4；参考 dental |
| 训练轮数 / batch | 200 / 2，是新训练入口默认值，不是论文披露参数 |
| 增强 | 旋转 ±5°、缩放 0.9–1.1、水平翻转；图像和标注同步变换 |
| 局部中心扰动 | crop 宽、高的 ±10%，仅训练使用 |
| 梯度裁剪 | 范数上限 5，可调 |
| 学习率 | 验证目标连续 10 轮未改善后乘 0.5 |

可使用 `--wsm-margin`、`--wsm-blur`、`--tsm-margin`、`--tsm-blur`、`--sigma`、
`--crop-margin`、`--crop-jitter` 等调整。重叠的多通道目标会归一化，避免出现负背景。
`--no-augment` 关闭所有训练增强和局部中心扰动；验证始终无随机增强。

新 Mask R-CNN 使用 torchvision ResNet50-FPN，输入为复制成三通道的灰度图，
归一化使用 torchvision 模型默认值。默认随机初始化；加 `--pretrained-detector` 才会
下载 COCO 初始权重。与原 GluonCV 的 ResNet50-v1b 网络和归一化细节不同，属于
现代训练实现，不能声称严格复现历史实验或直接使用其 `.params`。

## 4. 保存、续训和初始化

每个输出目录保存：

- `last.pth`：最后一轮完整状态，用于续训。
- `best.pth`：验证集最优轮的完整状态，可用于新推理适配器。
- `best_weights.pth`：最优网络的原始 `state_dict`，用于旧 U-Net 加载方式。
- `config.json`、`split.json`：参数与数据路径、文件 SHA256。
- `history.jsonl`：每轮训练损失、验证目标、学习率。
- `data_audit.json`：源数据数量、每个被排除样本的路径与原因、最终训练/验证数量。

续训命令需保留原训练参数，例如：

```powershell
python train.py --stage wsm --data-root "$dataRoot" --invalid-samples skip --output-dir saved_model/wsm --resume saved_model/wsm/last.pth --epochs 300
```

`--epochs 300` 表示总共训练到第 300 轮。断点包括优化器、调度器、最佳目标、随机数
及数据加载器状态；本次验证了 CPU 上分段续训与连续训练一致。不同硬件或 CUDA 算子
不保证逐位一致。恢复时会核对阶段、架构、目标参数、数据划分及内容等关键配置。
已有实验目录默认拒绝覆盖；恢复旧于当前日志的断点时，使用新的输出目录。

只加载权重开始新实验：

```powershell
python train.py --stage local-points --data-root "$dataRoot" --invalid-samples skip --output-dir saved_model/local-points-finetune --init-weights saved_model/local-points/best_weights.pth --lr 0.001
```

不要把原始权重传给 `--resume`；它们没有优化器状态，应使用 `--init-weights`。
数据格式适配前保存的旧完整断点也应使用 `--init-weights` 开始新实验，避免在数据划分
或读取规则变化后沿用优化器；新断点会核对 label format、异常策略、划分参数及 CSV 行号。
新完整断点使用 `torch.load(..., weights_only=True)` 读取；只加载自己信任的模型文件。

## 5. 验证目标

热图网络：独立验证集的平均 BCE+Focal 损失，越低越好。
检测器：按椎体 tag 匹配最高分检测，计算每个真实实例的二值 mask IoU；未检测到的
tag 计 0，概率阈值为 0.5、检测分数下限为 0.05。`val_mask_iou` 越高越好，
`val_objective = 1 - val_mask_iou` 越低越好。
该指标用于调试和选模，不等同于 COCO AP，也不是最终 SMAPE。

## 6. 对接原推理流程

U-Net 的参数名和通道布局保持兼容。旧代码的 `load_state_dict(torch.load(path))`
应指向对应阶段的 `best_weights.pth`，并确认 B 的 U-Net/ResUNet 选择一致。
移除了 U-Net 两个模块导入时修改全局 Tensor 类型的语句，没有改变网络参数结构。

新检测器示例：

```python
from predict_detector import TorchvisionSpineDetector

detector = TorchvisionSpineDetector(
    "saved_model/maskrcnn/best.pth",
    device="cuda",
    wsm_checkpoint="saved_model/wsm/best_weights.pth",
)
# image: HxW 或 1xHxW 灰度 numpy 数组，返回坐标仍在原输入分辨率
masks, boxes, centers, tags = detector.forward(image)
```

返回值对应旧检测器的四项输出；tags 转回 0–16，并按框中心纵坐标排序。
如果在旧流水线中替换检测器，需要更改实例化和调用代码：旧流水线已乘过 WSM 时
调用 `detector.forward(image, already_masked=True)`，避免重复乘遮罩。
新适配器不接受旧类的额外位置参数，如可视化标志和图像 ID。旧 MXNet 推理脚本
没有自动改成使用新检测器，历史 `.params` 路径也没有覆盖。
若仅有检测器的原始 `best_weights.pth`，还需显式传入训练时的 `size`；建议使用
带有配置的 `best.pth`，并显式提供训练所用 A 的权重。

## 7. 自检

```powershell
python -m unittest discover -s tests -v
```

11 项回归测试使用自动生成的图像与 68 点标注，包含真实网络反向传播、原输入尺寸的前向
检查、Windows 子进程数据加载、遮罩分支、权重读写、空检测结果以及断点续训一致性。
还覆盖原生目录、CSV 按文件名配对、归一化坐标、异常标注审计和病例组划分；
回归测试不下载预训练权重，不依赖患者数据。

正式数据的少量批次检查，可给任意训练命令增加：

```text
--epochs 1 --max-steps 2 --max-val-steps 1 --output-dir saved_model/debug-run
```

`--max-val-steps` 会截断验证，`--proposal-limit` 会减少检测 proposal，
`--allow-unmasked-detector` 会绕过论文的 WSM 输入遮罩；这些是调试/消融选项，
正式实验应移除。正式实验还应在独立测试集评估完整推理流水线的最终指标。
