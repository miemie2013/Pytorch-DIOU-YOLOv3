#! /usr/bin/env python
# coding=utf-8
#================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-11 16:31:57
#   Description : pytorch_yolov3
#                 导出为仓库 https://github.com/miemie2013/Keras-DIOU-YOLOv3
#                 所使用的的Keras模型。
#
#================================================================

import torch
import keras
import keras.layers as layers

from model.darknet_yolo_pt import Darknet



import platform
print(torch.cuda.is_available())
sysstr = platform.system()
print(torch.__version__)
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False





def conv2d_unit(x, filters, kernels, strides=1, padding='same'):
    x = layers.Conv2D(filters, kernels,
               padding=padding,
               strides=strides,
               use_bias=False,
               activation='linear',
               kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(inputs, filters):
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = layers.add([inputs, x])
    x = layers.Activation('linear')(x)
    return x

def stack_residual_block(inputs, filters, n):
    x = residual_block(inputs, filters)
    for i in range(n - 1):
        x = residual_block(x, filters)
    return x







def find(base_model, conv2d_name, batch_normalization_name):
    i1, i2 = -1, -1
    for i in range(len(base_model.layers)):
        if base_model.layers[i].name == conv2d_name:
            i1 = i
        if base_model.layers[i].name == batch_normalization_name:
            i2 = i
    return i1, i2


def aaaaaaa(conv, bn, cccccccc):
    conv2, bn2 = cccccccc.conv, cccccccc.bn

    w = conv2.weight.data.numpy()
    y = bn2.weight.data.numpy()
    b = bn2.bias.data.numpy()
    m = bn2.running_mean.data.numpy()
    v = bn2.running_var.data.numpy()

    w = w.transpose(2, 3, 1, 0)

    conv.set_weights([w])
    bn.set_weights([y, b, m, v])



def aaaaaaa2(conv, cccccccc):
    w = cccccccc.weight.data.numpy()
    b = cccccccc.bias.data.numpy()

    w = w.transpose(2, 3, 1, 0)

    conv.set_weights([w, b])

def bbbbbbbbbbb(base_model, stack_residual_block, start_index):
    i = start_index
    for residual_block in stack_residual_block.sequential:
        conv1 = residual_block.conv1
        conv2 = residual_block.conv2

        i1, i2 = find(base_model, 'conv2d_%d'%(i, ), 'batch_normalization_%d'%(i, ))
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], conv1)
        i1, i2 = find(base_model, 'conv2d_%d'%(i+1, ), 'batch_normalization_%d'%(i+1, ))
        aaaaaaa(base_model.layers[i1], base_model.layers[i2], conv2)
        i += 2



if __name__ == '__main__':
    num_classes = 20


    self = Darknet(num_classes, initial_filters=8)
    self.load_state_dict(torch.load('aaaa_bgr.pt'))
    self.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.




    # pattern取0时，初始卷积核个数
    initial_filters = 8

    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8
    i512 = i32 * 16
    i1024 = i32 * 32

    # 多尺度训练
    # inputs = layers.Input(shape=(None, None, 3))
    inputs = layers.Input(shape=(416, 416, 3))

    ''' darknet53部分，所有卷积层都没有偏移use_bias=False '''
    x = conv2d_unit(inputs, i32, (3, 3))

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i64, (3, 3), strides=2, padding='valid')
    x = stack_residual_block(x, i32, n=1)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i128, (3, 3), strides=2, padding='valid')
    x = stack_residual_block(x, i64, n=2)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = conv2d_unit(x, i256, (3, 3), strides=2, padding='valid')
    act11 = stack_residual_block(x, i128, n=8)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(act11)
    x = conv2d_unit(x, i512, (3, 3), strides=2, padding='valid')
    act19 = stack_residual_block(x, i256, n=8)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(act19)
    x = conv2d_unit(x, i1024, (3, 3), strides=2, padding='valid')
    act23 = stack_residual_block(x, i512, n=4)
    ''' darknet53部分结束，余下部分不再有残差块stack_residual_block() '''

    ''' 除了y1 y2 y3之前的1x1卷积有偏移，所有卷积层都没有偏移use_bias=False '''
    x = conv2d_unit(act23, i512, (1, 1), strides=1)
    x = conv2d_unit(x, i1024, (3, 3), strides=1)
    x = conv2d_unit(x, i512, (1, 1), strides=1)
    x = conv2d_unit(x, i1024, (3, 3), strides=1)
    lkrelu57 = conv2d_unit(x, i512, (1, 1), strides=1)

    x = conv2d_unit(lkrelu57, i1024, (3, 3), strides=1)
    y1 = layers.Conv2D(3 * (num_classes + 5), (1, 1))(x)
    # y1 = layers.Reshape((13, 13, 3, (num_classes + 5)), name='y1_r')(x)

    x = conv2d_unit(lkrelu57, i256, (1, 1), strides=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, act19])

    x = conv2d_unit(x, i256, (1, 1), strides=1)
    x = conv2d_unit(x, i512, (3, 3), strides=1)
    x = conv2d_unit(x, i256, (1, 1), strides=1)
    x = conv2d_unit(x, i512, (3, 3), strides=1)
    lkrelu64 = conv2d_unit(x, i256, (1, 1), strides=1)

    x = conv2d_unit(lkrelu64, i512, (3, 3), strides=1)
    y2 = layers.Conv2D(3 * (num_classes + 5), (1, 1))(x)
    # y2 = layers.Reshape((26, 26, 3, (num_classes + 5)), name='y2_r')(x)

    x = conv2d_unit(lkrelu64, i128, (1, 1), strides=1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, act11])

    x = conv2d_unit(x, i128, (1, 1), strides=1)
    x = conv2d_unit(x, i256, (3, 3), strides=1)
    x = conv2d_unit(x, i128, (1, 1), strides=1)
    x = conv2d_unit(x, i256, (3, 3), strides=1)
    x = conv2d_unit(x, i128, (1, 1), strides=1)
    x = conv2d_unit(x, i256, (3, 3), strides=1)
    y3 = layers.Conv2D(3 * (num_classes + 5), (1, 1))(x)
    # y3 = layers.Reshape((52, 52, 3, (num_classes + 5)), name='y3_r')(x)
    base_model = keras.models.Model(inputs=inputs, outputs=[y1, y2, y3])
    base_model.summary()

    # 复制参数
    print()


    i1, i2 = find(base_model, 'conv2d_1', 'batch_normalization_1')
    # self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv1)

    dd = 2
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv2)


    # self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)
    bbbbbbbbbbb(base_model, self.stack_residual_block_1, start_index=3)




    dd = 5
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv3)
    # self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)
    bbbbbbbbbbb(base_model, self.stack_residual_block_2, start_index=6)



    dd = 10
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv4)
    # self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)
    bbbbbbbbbbb(base_model, self.stack_residual_block_3, start_index=11)



    dd = 27
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv5)
    # self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)
    bbbbbbbbbbb(base_model, self.stack_residual_block_4, start_index=28)



    dd = 44
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv6)
    # self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
    bbbbbbbbbbb(base_model, self.stack_residual_block_5, start_index=45)


    ''' darknet53部分结束 '''

    dd = 53
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv53 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv53)
    dd = 54
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv54 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv54)
    dd = 55
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv55 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv55)
    dd = 56
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv56 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv56)
    dd = 57
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv57 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv57)



    dd = 58
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv58 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv58)



    dd = 59
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv59 = torch.nn.Conv2d(i1024, 3*(num_classes + 5), kernel_size=(1, 1))
    aaaaaaa2(base_model.layers[i1], self.conv59)





    dd = 60
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv60 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv60)
    # self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')

    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv61 = Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv61)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv62 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv62)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv63 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv63)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv64 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv64)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv65 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv65)

    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-1, ))
    # self.conv66 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv66)



    dd = 67
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv67 = torch.nn.Conv2d(i512, 3*(num_classes + 5), kernel_size=(1, 1))
    aaaaaaa2(base_model.layers[i1], self.conv67)














    dd = 68
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv68 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv68)
    # self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv69 = Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv69)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv70 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv70)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv71 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv71)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv72 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv72)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv73 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv73)
    dd += 1
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd-2, ))
    # self.conv74 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
    aaaaaaa(base_model.layers[i1], base_model.layers[i2], self.conv74)



    dd = 75
    i1, i2 = find(base_model, 'conv2d_%d'%(dd, ), 'batch_normalization_%d'%(dd, ))
    # self.conv75 = torch.nn.Conv2d(i256, 3*(num_classes + 5), kernel_size=(1, 1))
    aaaaaaa2(base_model.layers[i1], self.conv75)




    base_model.save('qqq.h5')
    # keras.utils.vis_utils.plot_model(base_model, to_file='aaa.png', show_shapes=True)


