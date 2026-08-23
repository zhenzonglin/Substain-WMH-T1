Substain 多卷迁移包

1. 把transfer目录中的全部文件传到Ubuntu-20.04工作站。
2. 在分卷所在目录校验：sha256sum -c SHA256SUMS
3. 建立一个空的目标父目录，并把全部 *.tar.gz 依次解压到该父目录：
   for part in *.tar.gz; do tar -xzf "$part" -C /目标父目录; done
4. 所有分卷会共同组装成 /目标父目录/Substain。
5. 初始化迁移环境：bash /目标父目录/Substain/scripts/finalize_transfer.sh
6. 完整验证：bash /目标父目录/Substain/scripts/verify_transferred_project.sh

分卷已经直接展开WMH/T1环境，不需要拼接大文件。首次初始化只修正conda-pack路径，不访问网络。
原始BIDS、Lesion、历史archive、derivatives和源项目transfer目录均未进入迁移包。
