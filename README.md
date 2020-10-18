# 环境

## 安装Anaconda

安装python=3.7的Anaconda

  https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/  

## 修改源 

按该网址修改为清华源

 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/ 

## 安装PyTorch

``conda create pytorch torchvision -c pytorch ``

## 安装其它依赖包

`pip install opencv-python`

`pip install nibabel` 

`pip install tensorboardX`

`pip install medpy`

# 数据处理 

将LiTS数据集（三个压缩包，LITS-Challenge-Test-Data.zip、Training_Batch1.zip、Training_Batch2.zip）中的所有nii文件解压至`data/LITS`目录下，运行项目根目录下的`preprocessing.py`

`python preprocessing.py`

# 训练

数据处理结束后，运行`main.py`训练U-Net网络line 20-44为训练选项的设定

`python main.py`

训练过程生成的检查点保存在`checkpoints`目录下，用于后续的分割结果生成

训练过程的训练集、测试集损失曲线保存在`runs`目录下，安装tensorboard后在项目根目录下输入以下命令可查看

`tensorboard --logdir runs`

# 测试

选择合适的checkpoints并将其路径写入`test.py`的line 16，之后运行`test.py`

`python test.py`

会自动生成`results`目录，结果按照case保存，其中

`results/X/mask`为网络输出的预测结果

`results/X/overlay`为原图与预测结果的叠加图