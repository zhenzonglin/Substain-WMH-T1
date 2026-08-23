# 第三方许可与再分发边界

- WMH progression modeling：保留上游仓库声明与提交历史；使用前复核仓库许可证。
- WMH-SynthSeg：保留上游仓库声明与模型来源；引用其论文与项目页面。
- FreeSurfer SynthStrip：程序固定为v7.4.1提交`7eb846079b0dc0c92e8313205a3d2387b5c7a354`，随源码保留FreeSurfer Software License Agreement v1.0；模型v1按官方提供的MIT选项纳入，许可文本见`SynthStrip_MODEL_MIT.txt`。项目运行时副本明确标记了PyTorch兼容性修改。
- ANTs：Apache License 2.0，以官方发行版为准。
- GenMIND：CSV、KDE和列字典仅作为本项目的技术常模/离线复现资源；保留数据来源、论文引用和上游使用条款，未经确认不公开重打包。
- DLMUSE、NiChart_DLMUSE、DLICV：安装或使用即受 CBICA/宾夕法尼亚大学非商业科研软件协议约束。项目离线包只在本项目和已授权工作站之间转移，不作为公开软件或模型包再分发。
- 离线烟雾测试镜像：基于 NVIDIA CUDA 12.4.1 与 Ubuntu 20.04；只用于项目内完整性测试，分发前分别复核 NVIDIA 条款和镜像内各 Ubuntu 软件包许可证。
- FSL MNI152：V1.0保存1 mm与2 mm标准参考网格及来源哈希，仅用于空间审计；保留`resources/templates/fsl/COPYRIGHTS.txt`。

正式迁移前，操作者必须确认目标工作站和研究用途满足相应许可证；SHA256 完整性不等同于授权。
