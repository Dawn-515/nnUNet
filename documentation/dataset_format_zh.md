# nnU-Net 数据集格式

将您的数据引入 nnU-Net 的唯一方法是以特定格式存储它。由于 nnU-Net 起源于
[医学分割十项全能竞赛](http://medicaldecathlon.com/)（MSD），其数据集受到了很大启发，但后来
与 MSD 使用的格式有所不同（另请参见[此处](#如何使用十项全能竞赛数据集)）。

数据集由三个组件组成：原始图像、相应的分割图和指定某些元数据的 dataset.json 文件。

如果您正在从 nnU-Net v1 迁移，请阅读[此处](#如何使用-nnu-net-v1-任务)以转换您现有的任务。

## 训练案例是什么样子的？

每个训练案例都与一个标识符（该案例的唯一名称）相关联。nnU-Net 使用此标识符
将图像与正确的分割连接起来。

训练案例由图像及其相应的分割组成。

**图像**是复数，因为 nnU-Net 支持任意多的输入通道。为了尽可能灵活，
nnU-Net 要求每个输入通道都存储在单独的图像中（唯一的例外是 RGB 自然
图像）。所以这些图像可以是例如 T1 和 T2 MRI（或者您想要的任何其他内容）。不同的输入
通道必须具有相同的几何形状（相同的形状、间距（如果适用）等），并且
必须是共同注册的（如果适用）。nnU-Net 通过文件末尾的 FILE_ENDING（一个四位数整数）来识别输入通道：
因此，图像文件必须遵循以下命名约定：{CASE_IDENTIFIER}_{XXXX}.{FILE_ENDING}。
其中，XXXX 是 4 位的模态/通道标识符（对于每个模态/通道应该是唯一的，例如，T1 的 "0000"，
T2 MRI 的 "0001"，...），FILE_ENDING 是您的图像格式使用的文件扩展名（.png, .nii.gz, ...）。请参见下面的具体示例。
dataset.json 文件在 'channel_names' 键中连接通道名称与通道标识符（有关详细信息，请参见下文）。

附注：通常，每个通道/模态需要存储在单独的文件中，并使用 XXXX 通道标识符访问。
例外是自然图像（RGB；.png）,其中三个颜色通道都可以存储在一个文件中（参见
[道路分割](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py) 数据集作为示例）。

**分割**必须与其相应的图像共享相同的几何形状（相同的形状等）。分割是
整数图，每个值代表一个语义类别。背景必须为 0。如果没有背景，那么
不要将标签 0 用于其他内容！您的语义类别的整数值必须是连续的（0, 1, 2, 3,
...）。当然，并非所有标签都必须在每个训练案例中出现。分割保存为 {CASE_IDENTIFER}.{FILE_ENDING}。

在训练案例中，所有图像几何形状（输入通道、相应的分割）必须匹配。在训练案例之间，它们当然可以不同。nnU-Net 会处理这个问题。

重要提示：输入通道必须一致！具体来说，**所有图像需要相同顺序的相同输入通道，并且所有输入通道都必须每次都存在**。这对推理也是如此！

## 支持的文件格式

nnU-Net 期望图像和分割使用相同的文件格式！这些也将用于推理。目前，
因此不可能训练 .png 然后对 .jpg 运行推理。

nnU-Net V2 的一个重大变化是支持多种输入文件类型。不再需要将所有内容转换为 .nii.gz！
这是通过 `BaseReaderWriter` 抽象化图像和分割的输入和输出来实现的。nnU-Net
带有广泛的读取器和写入器集合，您甚至可以添加自己的读取器和写入器来支持您的数据格式！
参见[此处](../nnunetv2/imageio/readme.md)。

作为一个很好的额外好处，nnU-Net 现在也原生支持 2D 输入图像，您不再需要
处理转换为伪 3D nifti 的问题。呃，那太恶心了。

请注意，在内部（用于存储和访问预处理的图像），无论原始数据提供的格式如何，nnU-Net 都将使用自己的文件格式！这是出于性能原因。

默认情况下，支持以下文件格式：

- NaturalImage2DIO: .png, .bmp, .tif
- NibabelIO: .nii.gz, .nrrd, .mha
- NibabelIOWithReorient: .nii.gz, .nrrd, .mha. 此读取器会将图像重定向到 RAS！
- SimpleITKIO: .nii.gz, .nrrd, .mha
- Tiff3DIO: .tif, .tiff. 3D tif 图像！由于 TIF 没有存储间距信息的标准化方法，
nnU-Net 期望每个 TIF 文件都附有一个同名的 .json 文件，其中包含这些信息（参见
[此处](#datasetjson)）。

文件扩展名列表并不详尽，取决于后端支持的内容。例如，nibabel 和 SimpleITK
支持的不仅仅是这里给出的三种。这里给出的文件扩展名只是我们测试过的那些！

重要提示：nnU-Net 只能与使用无损（或无）压缩的文件格式一起使用！因为文件
格式是为整个数据集定义的（而不是单独为图像和分割定义，这可能是未来的一个待办事项），我们必须确保
没有会破坏分割图的压缩伪影。所以不要使用 .jpg 之类的格式！

## 数据集文件夹结构

数据集必须位于 `nnUNet_raw` 文件夹中（您要么在安装 nnU-Net 时定义，要么每次
打算运行 nnU-Net 命令时导出/设置！）。
每个分割数据集作为一个单独的"数据集"存储。数据集与数据集 ID 相关联，数据集 ID 是一个三位
整数和一个数据集名称（您可以自由选择）：例如，Dataset005_Prostate 的数据集名称为 'Prostate'，
数据集 ID 为 5。数据集在 `nnUNet_raw` 文件夹中的存储方式如下：

    nnUNet_raw/
    ├── Dataset001_BrainTumour
    ├── Dataset002_Heart
    ├── Dataset003_Liver
    ├── Dataset004_Hippocampus
    ├── Dataset005_Prostate
    ├── ...

在每个数据集文件夹内，预期的结构如下：

    Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs  # 可选
    └── labelsTr

添加您的自定义数据集时，请查看 [dataset_conversion](../nnunetv2/dataset_conversion) 文件夹，
并选择一个尚未使用的 ID。ID 001-010 是为医学分割十项全能竞赛保留的。

- **imagesTr** 包含属于训练案例的图像。nnU-Net 将使用此数据执行管道配置、交叉验证训练
以及查找后处理和最佳集成。
- **imagesTs**（可选）包含属于测试案例的图像。nnU-Net 不使用它们！这只是
您存储这些图像的便利位置。这是医学分割十项全能竞赛文件夹结构的残留部分。
- **labelsTr** 包含训练案例的真实分割图的图像。
- **dataset.json** 包含数据集的元数据。

[上面](#训练案例是什么样子的)介绍的方案导致以下文件夹结构。以
MSD 的第一个数据集 BrainTumour 为例。该数据集有四个输入通道：FLAIR (0000)、
T1w (0001)、T1gd (0002) 和 T2w (0003)。请注意，imagesTs 文件夹是可选的，不必存在。

    nnUNet_raw/Dataset001_BrainTumour/
    ├── dataset.json
    ├── imagesTr
    │   ├── BRATS_001_0000.nii.gz
    │   ├── BRATS_001_0001.nii.gz
    │   ├── BRATS_001_0002.nii.gz
    │   ├── BRATS_001_0003.nii.gz
    │   ├── BRATS_002_0000.nii.gz
    │   ├── BRATS_002_0001.nii.gz
    │   ├── BRATS_002_0002.nii.gz
    │   ├── BRATS_002_0003.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── BRATS_485_0000.nii.gz
    │   ├── BRATS_485_0001.nii.gz
    │   ├── BRATS_485_0002.nii.gz
    │   ├── BRATS_485_0003.nii.gz
    │   ├── BRATS_486_0000.nii.gz
    │   ├── BRATS_486_0001.nii.gz
    │   ├── BRATS_486_0002.nii.gz
    │   ├── BRATS_486_0003.nii.gz
    │   ├── ...
    └── labelsTr
        ├── BRATS_001.nii.gz
        ├── BRATS_002.nii.gz
        ├── ...

这是 MSD 的第二个数据集的另一个例子，它只有一个输入通道：

    nnUNet_raw/Dataset002_Heart/
    ├── dataset.json
    ├── imagesTr
    │   ├── la_003_0000.nii.gz
    │   ├── la_004_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── la_001_0000.nii.gz
    │   ├── la_002_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── la_003.nii.gz
        ├── la_004.nii.gz
        ├── ...

记住：对于每个训练案例，所有图像必须具有相同的几何形状，以确保它们的像素数组对齐。还要
确保所有数据都是共同注册的！

另请参见[数据集格式推理](dataset_format_inference_zh.md)！

## dataset.json

dataset.json 包含 nnU-Net 训练所需的元数据。自版本 1 以来，我们大大减少了所需
字段的数量！

以下是 dataset.json 应该是什么样子的示例，以 MSD 的 Dataset005_Prostate 为例：

    { 
     "channel_names": {  # 以前称为 modalities
       "0": "T2", 
       "1": "ADC"
     }, 
     "labels": {  # 这里现在不同！
       "background": 0,
       "PZ": 1,
       "TZ": 2
     }, 
     "numTraining": 32, 
     "file_ending": ".nii.gz"
     "overwrite_image_reader_writer": "SimpleITKIO"  # 可选！如果未提供，nnU-Net 将自动确定 ReaderWriter
     }

channel_names 决定了 nnU-Net 使用的归一化。如果将通道标记为 'CT'，则将使用基于
前景像素中强度的全局归一化。如果是其他情况，将使用按通道 z-score 归一化。
有关更多详情，请参阅[我们的论文](https://www.nature.com/articles/s41592-020-01008-z)的方法部分。
nnU-Net v2 引入了更多可供选择的归一化方案，
并允许您定义自己的方案，更多信息请参见[此处](explanation_normalization_zh.md)。

相对于 nnU-Net v1 的重要变化：

- "modality"现在称为"channel_names"，以消除对医学图像的强烈偏见
- 标签的结构不同（名称 -> 整数，而不是整数 -> 名称）。这是支持[基于区域的训练](region_based_training_zh.md)所需的
- 添加了"file_ending"以支持不同的输入文件类型
- "overwrite_image_reader_writer" 可选！可用于指定应与此数据集一起使用的某个（自定义）ReaderWriter 类。如果未提供，nnU-Net 将自动确定 ReaderWriter
- "regions_class_order" 仅在[基于区域的训练](region_based_training_zh.md)中使用

有一个工具可以自动生成 dataset.json。您可以在
[此处](../nnunetv2/dataset_conversion/generate_dataset_json.py)找到它。
有关如何使用它，请查看我们在 [dataset_conversion](../nnunetv2/dataset_conversion) 中的示例。并阅读其文档！

如上所述，TIFF 文件需要包含间距信息的 json 文件。
对于 x 和 y 方向上对应于 7.6，z 方向上对应于 80 的 3D TIFF 堆栈，示例如下：

```
{
    "spacing": [7.6, 7.6, 80.0]
}
```

在数据集文件夹内，此文件（在本例中名为 `cell6.json`）将放置在以下文件夹中：

    nnUNet_raw/Dataset123_Foo/
    ├── dataset.json
    ├── imagesTr
    │   ├── cell6.json
    │   └── cell6_0000.tif
    └── labelsTr
        ├── cell6.json
        └── cell6.tif

## 如何使用 nnU-Net v1 任务

如果您正在从旧版 nnU-Net 迁移，请使用 `nnUNetv2_convert_old_nnUNet_dataset` 转换您现有的数据集！

迁移 nnU-Net v1 任务的示例：

```bash
nnUNetv2_convert_old_nnUNet_dataset /media/isensee/raw_data/nnUNet_raw_data_base/nnUNet_raw_data/Task027_ACDC Dataset027_ACDC 
```

使用 `nnUNetv2_convert_old_nnUNet_dataset -h` 获取详细使用说明。

## 如何使用十项全能竞赛数据集

参见 [convert_msd_dataset_zh.md](convert_msd_dataset_zh.md)

## 如何在 nnU-Net 中使用 2D 数据

现在原生支持 2D（耶！）。另请参见[此处](#支持的文件格式)以及此[脚本](../nnunetv2/dataset_conversion/Dataset120_RoadSegmentation.py)中的示例数据集。

## 如何更新现有数据集

更新数据集时，最佳做法是删除 `nnUNet_preprocessed/DatasetXXX_NAME` 中的预处理数据，
以确保重新开始。然后替换 `nnUNet_raw` 中的数据并重新运行 `nnUNetv2_plan_and_preprocess`。可选地，
还可以删除旧训练的结果。

# 示例数据集转换脚本

在 `dataset_conversion` 文件夹中（参见[此处](../nnunetv2/dataset_conversion)）有多个将数据集转换为 nnU-Net 格式的示例脚本。
这些脚本不能按原样运行（您需要打开它们并更改一些路径），但它们是您学习如何将自己的数据集转换为 nnU-Net 格式的极好示例。
只需选择最接近您的数据集作为起点。
数据集转换脚本列表会不断更新。如果您发现某些公开可用的数据集缺失，欢迎提交 PR 添加它！
