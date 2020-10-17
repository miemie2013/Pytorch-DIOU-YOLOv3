# Pytorch-DIOU-YOLOv3
<p align="center">
    <img width="100%" src="https://github.com/miemie2013/Keras-DIOU-YOLOv3/blob/master/weixin/diou.png" style="max-width:100%;">
    </a>
</p>
<p align="center">
    <img width="100%" src="https://github.com/miemie2013/Keras-DIOU-YOLOv3/blob/master/weixin/duibi.png" style="max-width:100%;">
    </a>
</p>

## 传送门

Keras版YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch版YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle完美复刻版版yolact: https://github.com/miemie2013/PaddlePaddle_yolact

Keras版YOLOv4: https://github.com/miemie2013/Keras-YOLOv4 (mAP 41%+)

Pytorch版YOLOv4: https://github.com/miemie2013/Pytorch-YOLOv4 (mAP 41%+)

Paddle版YOLOv4：https://github.com/miemie2013/Paddle-YOLOv4 (mAP 41%+)

PaddleDetection版SOLOv2: https://github.com/miemie2013/PaddleDetection-SOLOv2

Pytorch实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Pytorch-FCOS

Paddle实时版FCOS，跑得比YOLOv4快: https://github.com/miemie2013/Paddle-FCOS

Keras版CartoonGAN: https://github.com/miemie2013/keras_CartoonGAN

纯python实现一个深度学习框架: https://github.com/miemie2013/Pure_Python_Deep_Learning

Pytorch版PPYOLO: https://github.com/miemie2013/Pytorch-PPYOLO(mAP 44.8%)

## 更新日记

2020/01/13:初次见面

## 需要补充

冻结、解冻的代码，转rgb输入的脚本，其它稀奇古怪的东西

## 概述

Pytorch复现YOLOv3！使用DIOU loss训练。支持将模型导出为keras模型！
请查看
[ **`diou_loss的论文`**](https://arxiv.org/pdf/1911.08287.pdf)

参考了1个仓库：

https://github.com/YunYang1994/tensorflow-yolov3

这个仓库有很大一部分参考（用pytorch的api翻译）了YunYang1994的代码（label的填写以及损失函数部分），这里致敬大佬！
大部分为自己原创（我只是个搬砖的）。
YunYang1994的仓库训练出的模型很优秀，为了达到同等优秀的效果，所以损失函数部分参考了大佬仓库里的代码。


## 文件下载
在coco上的预训练模型yolo_bgr_mAP_47.pt，在release处下载。

coco2017数据集下载：

http://images.cocodataset.org/zips/train2017.zip 

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip 

http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip 

http://images.cocodataset.org/annotations/image_info_test2017.zip



## 仓库文件介绍

```
train.py            训练yolov3，用的是ciou loss。
2_pytorch2keras.py  将pytorch模型导出为keras模型。给兄弟仓库兄弟版：https://github.com/miemie2013/Keras-DIOU-YOLOv3使用。
demo.py             用pytorch模型进行预测。对视频进行预测的话需要解除注释。
eval.py             对pytorch模型评估。跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。


annotation/  存放训练集、验证集的注解文件。
data/        存放数据集物品类别名称文件（一行一个类别名称），类别名称最好不要有空格、斜杠、反斜杠，不然后面计算mAP时会报错。
images/      用于测试的图片，放在子目录test/下。预测输出在子目录res/下。
mAP/         对模型评估时产生的中间临时文件。
model/       存放yolov3算法后处理的脚本。
videos/      用于测试的视频，放在子目录test/下。
```

## 训练
使用train.py进行训练。train.py不支持命令行参数设置使用的数据集、超参数。
而是通过修改train.py源代码来进行更换数据集、更改超参数（减少冗余代码）。
1.如果你要使用自己的数据集训练，那么请修改
```
train_path = 'annotation/coco2017_train.txt'
val_path = 'annotation/coco2017_val.txt'
classes_path = 'data/coco_classes.txt'
```

注解文件的格式如下：
```
xxx/xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20 
xxx/xxx.jpg 48,240,195,371,11 8,12,352,498,14
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
# make sure that x_max < width and y_max < height
```
和YunYang1994的注解文件格式是完全一样的，这里再次致敬大佬！

2.本仓库有pattern=0、pattern=1、pattern=2这3种训练模式。
0-从头训练，1-读取模型继续训练（包括解冻），2-读取coco预训练模型训练
你只需要修改pattern的值即可指定训练模式。
然后在这3种模式的if-else分支下，你再指定批大小batch_size、学习率lr等超参数。

3.如果你决定从头训练一个模型（即pattern=0），而且你的显卡显存比较小，比如说只有6G。
又或者说你想训练一个小模型，因为你的数据集比较小。
那么你可以设置initial_filters为一个比较小的值，比如说8。
initial_filters会影响到后面的卷积层的卷积核个数（除了最后面3个卷积层的卷积核个数不受影响）。
yolov3的initial_filters默认是32，你调小initial_filters会使得模型变小，运算量减少，适合在小数据集上训练。


## 评估
训练完成后，运行eval.py对pytorch模型评估，跑完这个脚本后需要再跑mAP/main.py进行mAP的计算。


## 预测
运行demo.py。

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。


## 广告位招租
可联系微信wer186259
