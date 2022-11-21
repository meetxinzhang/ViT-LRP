from： https://blog.csdn.net/weixin_43002433/article/details/106225771

Windows：
1. 在Windows中下载Git
2. 打开数据集目录，Git bash分别运行data_prep_sh_files中的三个.sh文件，如sh data_prep_sh_files/trainprep.sh
3. (optional，实际标签是对应目录名的，也可按给出的开发文件中的label文件自行导入)如果使用ImageNet原始的label，则解压ILSVRC2012_devkit_t12.tar文件，解压出来如果是个名为'ILSVRC2012_devkit_t12'不带后缀的文件，继续解压这个文件即可；如果使用caffe版本的label，解压caffe_ilsvrc12.tar.gz文件。


Linux：
1. 打开终端，cd进入数据集目录
2. sh ./data_prep_sh_files/trainprep.sh
2. sh ./data_prep_sh_files/valprep.sh
3. sh ./data_prep_sh_files/testprep.sh


