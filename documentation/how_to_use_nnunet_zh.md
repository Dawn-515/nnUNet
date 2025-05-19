## **2024-04-18 更新：全新的残差编码器 UNet 预设可用！**

推荐的 nnU-Net 预设已经改变！查看[这里](resenc_presets_zh.md)了解如何使用它们！

## 如何在新数据集上运行 nnU-Net

给定某个数据集，nnU-Net 完全自动地配置一个与其属性匹配的完整分割流程。
nnU-Net 覆盖整个流程，从预处理到模型配置、模型训练、后处理
一直到集成。运行 nnU-Net 后，可以将训练好的模型应用于测试案例进行推理。

### 数据集格式

nnU-Net 期望数据集采用结构化格式。这种格式受到
[医学分割十项全能](http://medicaldecathlon.com/)数据结构的启发。请阅读
[这篇文档](dataset_format_zh.md)，了解如何设置与 nnU-Net 兼容的数据集。

**自版本 2 起，我们支持多种图像文件格式（.nii.gz、.png、.tif 等）！阅读 dataset_format
文档以了解更多信息！**

**来自 nnU-Net v1 的数据集可以通过运行 `nnUNetv2_convert_old_nnUNet_dataset INPUT_FOLDER
OUTPUT_DATASET_NAME` 转换到 V2。** 记住，v2 将数据集称为 DatasetXXX_Name（不是 Task），其中 XXX 是一个 3 位数字。
请提供旧任务的**路径**，而不仅仅是任务名称。nnU-Net V2 不知道 v1 任务在哪里！

### 实验规划和预处理

给定一个新数据集，nnU-Net 将提取数据集指纹（一组数据集特定属性，如
图像大小、体素间距、强度信息等）。这些信息用于设计三种 U-Net 配置。
每个流程都在数据集的自己预处理版本上运行。

运行指纹提取、实验规划和预处理的最简单方法是使用：

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

其中 `DATASET_ID` 是数据集 ID。我们建议在首次
运行此命令时使用 `--verify_dataset_integrity`。这将检查一些最常见的错误源！

您也可以通过指定 `-d 1 2 3 [...]` 一次处理多个数据集。如果您已经知道需要什么 U-Net 配置，
也可以使用 `-c 3d_fullres` 指定（在这种情况下，请确保调整 -np！）。有关
您可用的所有选项的更多信息，请运行 `nnUNetv2_plan_and_preprocess -h`。

nnUNetv2_plan_and_preprocess 将在您的 nnUNet_preprocessed 文件夹中创建一个以数据集命名的新子文件夹。
命令完成后，将有一个 dataset_fingerprint.json 文件以及一个 nnUNetPlans.json 文件供您查看
（如果您有兴趣！）。还会有包含 UNet 配置的预处理数据的子文件夹。

[可选]
如果您喜欢将事情分开，也可以按顺序使用 `nnUNetv2_extract_fingerprint`、`nnUNetv2_plan_experiment`
和 `nnUNetv2_preprocess`。

### 模型训练

#### 概述

您可以选择应该训练哪些配置（2d、3d_fullres、3d_lowres、3d_cascade_fullres）！如果您不知道
什么在您的数据上表现最好，只需运行所有这些配置，让 nnU-Net 确定最佳的那个。这完全取决于您！

nnU-Net 在训练案例上使用 5 折交叉验证训练所有配置。这是 1) 需要的，以便
nnU-Net 可以估计每个配置的性能，并告诉您应该将哪个用于您的
分割问题，2) 获得良好模型集成（平均这 5 个模型的输出进行预测）以提高性能的自然方法。

您可以影响 nnU-Net 用于 5 折交叉验证的拆分（参见[此处](manual_data_splits_zh.md)）。如果您
喜欢在所有训练案例上训练单个模型，这也是可能的（见下文）。

**请注意，并非所有数据集都会创建所有 U-Net 配置。在图像大小较小的数据集中，会省略 U-Net
级联（及其 3d_lowres 配置），因为全分辨率 U-Net 的补丁大小
已经覆盖输入图像的很大一部分。**

训练模型是使用 `nnUNetv2_train` 命令完成的。该命令的一般结构是：

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

UNET_CONFIGURATION 是标识所请求的 U-Net 配置的字符串（默认：2d、3d_fullres、3d_lowres、
3d_cascade_lowres）。DATASET_NAME_OR_ID 指定应该在哪个数据集上进行训练，FOLD 指定
训练 5 折交叉验证的哪个折。

nnU-Net 每 50 个 epoch 保存一个检查点。如果您需要继续先前的训练，只需在
训练命令中添加 `--c`。

重要提示：如果您打算使用 `nnUNetv2_find_best_configuration`（见下文），请添加 `--npz` 标志。这会使
nnU-Net 在最终验证期间保存 softmax 输出。这些是需要的。导出的 softmax
预测非常大，因此会占用大量磁盘空间，这就是为什么默认情况下不启用它。
如果您最初没有使用 `--npz` 标志，但现在需要 softmax 预测，只需使用以下命令重新运行验证：

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```

您可以通过使用 `-device DEVICE` 来指定 nnU-Net 应该使用的设备。DEVICE 只能是 cpu、cuda 或 mps。如果
您有多个 GPU，请使用 `CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...]` 选择 gpu id（需要设备为 cuda）。

有关其他选项，请参见 `nnUNetv2_train -h`。

### 2D U-Net

对于 FOLD 在 [0, 1, 2, 3, 4] 中，运行：

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```

### 3D 全分辨率 U-Net

对于 FOLD 在 [0, 1, 2, 3, 4] 中，运行：

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
```

### 3D U-Net 级联

#### 3D 低分辨率 U-Net

对于 FOLD 在 [0, 1, 2, 3, 4] 中，运行：

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD [--npz]
```

#### 3D 全分辨率 U-Net

对于 FOLD 在 [0, 1, 2, 3, 4] 中，运行：

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD [--npz]
```

**注意，级联的 3D 全分辨率 U-Net 需要低分辨率 U-Net 的五个折都
完成！**

训练好的模型将被写入 nnUNet_results 文件夹。每个训练都会获得一个自动生成的
输出文件夹名称：

nnUNet_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD

例如，对于 MSD 中的 Dataset002_Heart，这看起来像这样：

    nnUNet_results/
    ├── Dataset002_Heart
        │── nnUNetTrainer__nnUNetPlans__2d
        │    ├── fold_0
        │    ├── fold_1
        │    ├── fold_2
        │    ├── fold_3
        │    ├── fold_4
        │    ├── dataset.json
        │    ├── dataset_fingerprint.json
        │    └── plans.json
        └── nnUNetTrainer__nnUNetPlans__3d_fullres
             ├── fold_0
             ├── fold_1
             ├── fold_2
             ├── fold_3
             ├── fold_4
             ├── dataset.json
             ├── dataset_fingerprint.json
             └── plans.json

注意，这里没有 3d_lowres 和 3d_cascade_fullres，因为这个数据集没有触发级联。在每个
模型训练输出文件夹（每个 fold_x 文件夹）中，将创建以下文件：

- debug.json：包含用于训练此模型的蓝图和推断参数的摘要，以及
一堆附加的东西。不容易阅读，但对调试非常有用；-)
- checkpoint_best.pth：训练期间确定的最佳模型的检查点文件。除非您
明确告诉 nnU-Net 使用它，否则现在不会使用。
- checkpoint_final.pth：最终模型的检查点文件（训练结束后）。这是用于
验证和推理的文件。
- network_architecture.pdf（仅当安装了 hiddenlayer！）：一个包含网络架构图的 pdf 文档。
- progress.png：显示训练过程中的损失、伪 Dice、学习率和 epoch 时间。顶部是
训练期间的训练（蓝色）和验证（红色）损失图。还显示 Dice 的近似值（绿色）以及
其移动平均线（绿色虚线）。此近似值是前景类的平均 Dice 分数。
**它需要非常（！）谨慎地看待**，因为它是在每个 epoch 结束时从验证数据中随机抽取的补丁上计算的，
并且用于 Dice 计算的 TP、FP 和 FN 的聚合将补丁视为
它们都来自同一卷（'全局 Dice'；我们不计算每个验证案例的 Dice，然后
在所有案例上平均，而是假设只有一个验证案例，我们从中采样补丁）。原因是
'全局 Dice' 在训练期间易于计算，并且对于评估模型是否正在训练仍然非常有用。
适当的验证需要太长时间才能在每个 epoch 完成。它在训练结束时运行。
- validation：在这个文件夹中是训练完成后预测的验证案例。这里的 summary.json 文件
包含验证指标（在文件开头提供了所有案例的平均值）。如果设置了 `--npz`，那么
压缩的 softmax 输出（保存为 .npz 文件）也在这里。

在训练期间，观察进度通常很有用。因此，我们建议您在运行第一次训练时查看生成的
progress.png。它将在每个 epoch 后更新。

训练时间主要取决于 GPU。我们推荐用于训练的最小 GPU 是 Nvidia RTX 2080ti。使用
该 GPU，所有网络训练需要不到 2 天的时间。请参考我们的[基准](benchmarking_zh.md)，查看您的系统是否
按预期执行。

### 使用多个 GPU 进行训练

如果有多个 GPU 可用，最好的使用方法是同时训练多个 nnU-Net，每个
GPU 上一个。这是因为数据并行性从不完美线性缩放，特别是对于 nnU-Net 使用的小型网络。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # 在 GPU 0 上训练
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # 在 GPU 1 上训练
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # 在 GPU 2 上训练
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # 在 GPU 3 上训练
CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # 在 GPU 4 上训练
...
wait
```

**重要：第一次运行训练时，nnU-Net 会将预处理的数据提取到未压缩的 numpy
数组中，以提高速度！在启动同一配置的多个训练之前，必须完成此操作！
在第一个训练开始使用 GPU 之前，请等待启动后续折！根据
数据集大小和您的系统，这最多应该只需要几分钟。**

如果您坚持运行 DDP 多 GPU 训练，我们也提供了支持：

`nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X`

再次注意，这将比在单独的 GPU 上运行单独的训练慢。DDP 只有在您手动
干预了 nnU-Net 配置并训练更大的模型，使用更大的补丁和/或批量大小时才有意义！

使用 `-num_gpus` 时需要注意：

1) 如果您使用 2 个 GPU 进行训练，但系统中有更多 GPU，您需要通过
CUDA_VISIBLE_DEVICES=0,1（或者您的 ID 是什么）指定应该使用哪些 GPU。
2) 您不能指定比小批量中的样本数更多的 GPU。如果批量大小是 2，则 2 个 GPU 是最大值！
3) 确保您的批量大小可以被您使用的 GPU 数量整除，否则您将无法充分利用您的硬件。

与旧版 nnU-Net 不同，DDP 现在完全没有麻烦。请享用！

### 自动确定最佳配置

一旦所需的配置训练完成（完整交叉验证），您可以告诉 nnU-Net 自动为您
确定最佳组合：

```commandline
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
```

这里的 `CONFIGURATIONS` 是您想要探索的配置列表。默认情况下，启用集成，
这意味着 nnU-Net 将生成所有可能的集成组合（每个集成 2 个配置）。这需要
存在包含验证集预测概率的 .npz 文件（使用带有
`--npz` 标志的 `nnUNetv2_train`，参见上文）。您可以通过设置 `--disable_ensembling` 标志来禁用集成。

有关更多选项，请参见 `nnUNetv2_find_best_configuration -h`。

nnUNetv2_find_best_configuration 还会自动确定应该使用的后处理。
nnU-Net 中的后处理只考虑删除预测中所有组件，只保留最大的组件（前景与背景之间执行一次，
对每个标签/区域执行一次）。

完成后，该命令会在控制台上打印您需要运行的确切命令，以进行预测。它
还会在 `nnUNet_results/DATASET_NAME` 文件夹中为您创建两个文件供您检查：

- `inference_instructions.txt` 再次包含您需要用于预测的确切命令
- `inference_information.json` 可以检查所有配置和集成的性能，以及
后处理的效果和一些调试信息。

### 运行推理

记住，位于输入文件夹中的数据必须具有与您训练模型的数据集相同的文件扩展名，
并且必须遵循 nnU-Net 对图像文件的命名方案（参见[数据集格式](dataset_format_zh.md)和
[推理数据格式](dataset_format_inference_zh.md)！）

`nnUNetv2_find_best_configuration`（参见上文）将在终端中打印一个包含您需要使用的推理命令的字符串。
运行推理的最简单方法是直接使用这些命令。

如果您希望手动指定用于推理的配置，请使用以下命令：

#### 运行预测

对于每个所需的配置，运行：

```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

仅当您打算使用集成时才指定 `--save_probabilities`。`--save_probabilities` 将使命令保存预测的
概率以及预测的分割掩码，这需要大量磁盘空间。

请为每个配置选择一个单独的 `OUTPUT_FOLDER`！

请注意，默认情况下，推理将使用交叉验证中的所有 5 个折作为集成。我们非常
强烈建议您使用所有 5 个折。因此，在运行推理之前，必须训练所有 5 个折。

如果您希望使用单个模型进行预测，请训练 `all` 折，并在 `nnUNetv2_predict`
中使用 `-f all` 指定它。

#### 集成多个配置

如果您希望集成多个预测（通常来自不同的配置），您可以使用以下命令：

```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -np NUM_PROCESSES
```

您可以指定任意数量的文件夹，但请记住，每个文件夹都需要包含由
`nnUNetv2_predict` 生成的 npz 文件。同样，`nnUNetv2_ensemble -h` 将告诉您有关其他选项的更多信息。

#### 应用后处理

最后，将先前确定的后处理应用于（集成的）预测：

```commandline
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

`nnUNetv2_find_best_configuration`（或其生成的 `inference_instructions.txt` 文件）将告诉您在哪里找到
后处理文件。如果没有，您可以在结果文件夹中查找它（它创造性地命名为
`postprocessing.pkl`）。如果您的源文件夹来自集成，您还需要指定一个 `-plans_json` 文件和
一个应该使用的 `-dataset_json` 文件（对于单一配置预测，这些文件会自动从
各自的训练中复制）。您可以从任何集成成员中选择这些文件。

## 如何使用预训练模型运行推理

参见[此处](run_inference_with_pretrained_models_zh.md)

## 如何部署和使用您的预训练模型运行推理

为了便于在不同计算机上使用预训练模型进行推理，请按照以下简化步骤操作：

1. 导出模型：利用 `nnUNetv2_export_model_to_zip` 函数将您的训练模型打包成 .zip 文件。此文件将包含所有必要的模型文件。
2. 转移模型：将 .zip 文件转移到将进行推理的目标计算机。
3. 导入模型：在新电脑上，使用 `nnUNetv2_install_pretrained_model_from_zip` 从 .zip 文件加载预训练模型。
请注意，两台计算机都必须安装 nnU-Net 及其所有依赖项，以确保模型的兼容性和功能。

[//]: # (## 示例)

[//]: # ()
[//]: # (为了帮助您入门，我们编制了两个简单易懂的示例：)

[//]: # (- 在 Hippocampus 数据集上使用 3d 全分辨率 U-Net 运行训练。参见[此处]&#40;documentation/training_example_Hippocampus.md&#41;。)

[//]: # (- 在 Prostate 数据集上使用 nnU-Net 的预训练模型运行推理。参见[此处]&#40;documentation/inference_example_Prostate.md&#41;。)

[//]: # ()
[//]: # (可用性不够好？请告诉我们！)
