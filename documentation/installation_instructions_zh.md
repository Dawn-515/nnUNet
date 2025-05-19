# 系统要求

## 操作系统

nnU-Net 已在 Linux（Ubuntu 18.04、20.04、22.04；centOS、RHEL）、Windows 和 MacOS 上进行了测试！它应该可以开箱即用！

## 硬件要求

我们支持 GPU（推荐）、CPU 和 Apple M1/M2 作为设备（目前 Apple mps 尚未实现 3D 卷积，因此在这些设备上您可能需要使用 CPU）。

### 训练的硬件要求

我们建议您使用 GPU 进行训练，因为在 CPU 或 MPS（Apple M1/M2）上训练将花费非常长的时间。
训练需要至少 10 GB 的 GPU（流行的非数据中心选项有 RTX 2080ti、RTX 3080/3090 或 RTX 4080/4090）。
我们还建议使用强大的 CPU 配合 GPU。6 核（12 线程）是最低要求！CPU 要求主要与数据增强有关，
并随输入通道和目标结构的数量而变化。此外，GPU 越快，CPU 也应该越好！

### 推理的硬件要求

同样，我们建议使用 GPU 进行预测，因为这比其他选项要快得多。但是，
在 CPU 和 MPS（Apple M1/M2）上的推理时间通常仍然可以接受。如果使用 GPU，应至少有 4 GB 的可用（未使用）VRAM。

### 硬件配置示例

训练工作站配置示例：

- CPU：Ryzen 5800X - 5900X 或 7900X 会更好！我们尚未测试英特尔 Alder/Raptor lake，但它们可能也能正常工作。
- GPU：RTX 3090 或 RTX 4090
- RAM：64GB
- 存储：SSD（M.2 PCIe Gen 3 或更好！）

训练服务器配置示例：

- CPU：2x AMD EPYC7763，总计 128C/256T。对于 A100 等快速 GPU，强烈建议每 GPU 配备 16C！
- GPU：8xA100 PCIe（价格/性能优于 SXM 变体 + 它们使用更少的功率）
- RAM：1 TB
- 存储：本地 SSD 存储（PCIe Gen 3 或更好）或超高速网络存储

（nnU-net 默认每次训练使用一个 GPU。服务器配置可以同时运行最多 8 个模型训练）

### 为数据增强设置正确的工作进程数量（仅限训练）

请注意，您需要根据 CPU/GPU 比例手动设置 nnU-Net 用于数据增强的进程数。对于上述服务器（8 个 GPU 的 256 个线程），
一个好的值是 24-30。您可以通过设置 `nnUNet_n_proc_DA` 环境变量来实现（`export nnUNet_n_proc_DA=XX`）。
建议值（假设使用具有良好 IPC 的最新 CPU）：RTX 2080 ti 为 10-12，RTX 3090 为 12，
RTX 4090 为 16-18，A100 为 28-32。最佳值可能会根据输入通道/模态数量和类别数量而有所不同。

# 安装说明

我们强烈建议您在虚拟环境中安装 nnU-Net！Pip 或 anaconda 都可以。如果您选择
从源代码编译 PyTorch（见下文），则需要使用 conda 而不是 pip。

使用较新版本的 Python！3.9 或更新版本保证可以工作！

**nnU-Net v2 可以与 nnU-Net v1 共存！两者可以同时安装。**

1) 按照其网站上的说明安装 [PyTorch](https://pytorch.org/get-started/locally/)（conda/pip）。请
安装支持您的硬件（cuda、mps、cpu）的最新版本。
**不要在没有正确安装 PYTORCH 的情况下直接 `pip install nnunetv2`**。为了获得最高速度，请考虑
[自行编译 pytorch](https://github.com/pytorch/pytorch#from-source)（仅限有经验的用户！）。
2) 根据您的用例安装 nnU-Net：
    1) 用作**标准化基线**、**开箱即用的分割算法**或用于**使用预训练模型进行推理**：

       ```pip install nnunetv2```

    2) 用作集成**框架**（这将在您的计算机上创建 nnU-Net 代码的副本，以便您根据需要修改）：

          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```

3) nnU-Net 需要知道您打算保存原始数据、预处理数据和训练模型的位置。为此，您需要
   设置一些环境变量。请按照[此处](setting_up_paths_zh.md)的说明进行操作。
4) （可选）安装 [hiddenlayer](https://github.com/waleedka/hiddenlayer)。hiddenlayer 使 nnU-net 能够生成
   它生成的网络拓扑图（参见[模型训练](how_to_use_nnunet_zh.md#model-training)）。
要安装 hiddenlayer，
   运行以下命令：

    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
    ```

安装 nnU-Net 将向您的终端添加几个新命令。这些命令用于运行整个 nnU-Net
管道。您可以从系统上的任何位置执行它们。所有 nnU-Net 命令都有前缀 `nnUNetv2_`，便于
识别。

请注意，这些命令只是执行 python 脚本。如果您在虚拟环境中安装了 nnU-Net，
在执行命令时必须激活此环境。您可以通过查看 [pyproject.toml](../pyproject.toml) 文件中的 project.scripts 来查看执行了哪些脚本/函数。

所有 nnU-Net 命令都有 `-h` 选项，提供有关如何使用它们的信息。
