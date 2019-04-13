# 描述
使用CNN 迁移学习实现[Kaggle猫狗图像分类挑战](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)，代码基于[TF Slim](https://github.com/tensorflow/models/tree/master/research/slim)。完整流程描述参见 https://zhuanlan.zhihu.com/p/62317034。

## 使用方法

### 下载数据集
数据集下载地址：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

### 数据预处理
#### 清除异常数据
训练图片中有些不是猫狗图片，会影响训练，删除此部分图片。
#### 抽取验证集
从 train 目录随机抽取部分验证集数据到 validation 目录作为验证集。
### 下载预训练模型 
```
mkdir pre_train_model
cd pre_train_model
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
```

### 目录结构
```
--input  
   |__train 
   |__validation
   |__test
--logs
--pre_train_model
```

### 训练
``` bash
python train.py --model=inception_resnet_v2 --epochs=5 --learning_rate=0.0001 --steps=10000 --decay_steps=100000 --decay_rate=0.996
```
### 预测
```
python predict.py model=inception_resnet_v2 --image_dir="图片文件夹路径" --checkpoint="训练阶段保存的模型检查点"
```

## 依赖包
- python 2.7.14
- Tensorflow 1.3  
