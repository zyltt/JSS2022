import copy

import numpy as np
import tensorflow as tf
import tensorflow.keras.models as models



class Activation_Layer(tf.keras.Model):
    def __init__(self,type):
        super(Activation_Layer, self).__init__()
        self.type = type
    def call(self,x):
        if self.type == "relu":
            return tf.nn.relu(x)
        elif self.type == "sigmoid":
            return tf.nn.sigmoid(x)
        elif self.type == "tanh":
            return tf.nn.tanh(x)
        elif self.type == "leakyrelu":
            return tf.nn.leaky_relu(x)
        elif self.type == "prelu":
            #mindspore在cpu下不支持prelu，因此就不进行计算了
            return x
        elif self.type == "elu":
            return tf.nn.elu(x)

class Cell(tf.keras.Model):
    def __init__(self,in_channel,final_module,channels,activation_type):
        super(Cell, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
        self.activation_layer = Activation_Layer(activation_type)
    def call(self, inputs):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(inputs)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            final_point = eachOperation.toIndex
            fromIndex = eachOperation.fromIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # in_channel表示操作的入通道数
            operator_in_channel = self.channels[fromIndex] * self.in_channel
            operator = eachOperation.operator
            # indentity
            if operator == -1:
                # print("tensorflow执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = copy.deepcopy(tensors[fromIndex])
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = tf.concat([tensors[toIndex], input], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("tensorflow执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((1, 1,operator_in_channel , self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=True)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((1, 1, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=True)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("tensorflow执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel的倍数
                    filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    thisresult = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1, 1, 1, 1],
                                                        padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    thisresult = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1, 1, 1, 1],
                                                        padding='SAME')

                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                # 为了debug,先把这两步注释掉。
                # tensors[toIndex] = tf.layers.batch_normalization(tensors[toIndex])
                # tensors[toIndex] = tf.nn.relu(tensors[toIndex])
            elif operator == 3:
                # print("tensorflow执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 各filter参数顺序:H、W、InChannel、OutChannel
                    depthwise_filter = tf.ones((3, 3, operator_in_channel, 1), tf.float32)
                    pointwise_filter = tf.constant(value=1.0, shape=[1, 1, operator_in_channel, self.in_channel],
                                                   dtype=tf.float32)
                    tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                        pointwise_filter=pointwise_filter,
                                                        strides=[1, 1, 1, 1], padding='SAME')

                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(tempresult, axes=[0, 1, 2], keepdims=False)
                    tempresult = tf.nn.batch_normalization(tempresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    tempresult = self.activation_layer(tempresult)
                    tensors[toIndex] = copy.deepcopy(tempresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    depthwise_filter = tf.ones((3, 3, operator_in_channel, 1), dtype=tf.float32)
                    pointwise_filter = tf.ones((1, 1, operator_in_channel, self.in_channel), dtype=tf.float32)
                    tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                        pointwise_filter=pointwise_filter,
                                                        strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(tempresult, axes=[0, 1, 2], keepdims=False)
                    tempresult = tf.nn.batch_normalization(tempresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    tempresult = self.activation_layer(tempresult)
                    temp = tf.concat([tensors[toIndex], tempresult], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            elif operator == 4:
                # print("tensorflow执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.max_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            elif operator == 5:
                # print("tensorflow执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.avg_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.avg_pool2d(input=input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("tensorflow执行了操作6", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((3, 3, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, operator_in_channel, self.in_channel), dtype=tf.float32)
                    thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator==7:
                # print("tensorflow执行了操作6", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:H、W、InChannel、OutChannel
                    filter = tf.ones((3, 3, self.in_channel, operator_in_channel), dtype=tf.float32)
                    inputshape = tf.shape(input)
                    output_shape = [inputshape[0],inputshape[1],inputshape[2],self.in_channel]
                    thisresult = tf.nn.conv2d_transpose(input=input, filters=filter,output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = copy.deepcopy(thisresult)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    filter = tf.ones((3, 3, self.in_channel, operator_in_channel), dtype=tf.float32)
                    inputshape = tf.shape(input)
                    output_shape = [inputshape[0],inputshape[1],inputshape[2],self.in_channel]
                    thisresult = tf.nn.conv2d_transpose(input=input, filters=filter,output_shape=output_shape, strides=[1, 1, 1, 1], padding='SAME')
                    # 激活函数和归一化
                    thismean, thisvariance = tf.nn.moments(thisresult, axes=[0, 1, 2], keepdims=False)
                    thisresult = tf.nn.batch_normalization(thisresult, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    thisresult = self.activation_layer(thisresult)
                    temp = tf.concat([tensors[toIndex], thisresult], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #relu
            elif operator == 8:
                # print("tensorflow执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.relu(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.relu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #sigmoid
            elif operator == 9:
                # print("tensorflow执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.sigmoid(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.sigmoid(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #tanh
            elif operator == 10:
                # print("tensorflow执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.tanh(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.tanh(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #relu
            elif operator == 11:
                # print("tensorflow执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.leaky_relu(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.leaky_relu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #prelu
            elif operator == 12:
                # print("tensorflow执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = input
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = input
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #elu
            elif operator == 13:
                # print("tensorflow执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = tf.nn.elu(input)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    result = tf.nn.elu(input)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #batchnorm
            elif operator == 14:
                # print("tensorflow执行了了操作14 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    thismean, thisvariance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=True)
                    result = tf.nn.batch_normalization(input, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    tensors[toIndex] = copy.deepcopy(result)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    thismean, thisvariance = tf.nn.moments(input, axes=[0, 1, 2], keepdims=True)
                    result = tf.nn.batch_normalization(input, mean=thismean, variance=thisvariance,
                                                           offset=None, scale=None, variance_epsilon=1e-5)
                    temp = tf.concat([tensors[toIndex], result], 3)
                    tensors[toIndex] = copy.deepcopy(temp)
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        return copy.deepcopy(tensors[final_point])





class TFNet(tf.keras.Model):
    def __init__(self, channels, final_module, in_channel, activation_type):
        super(TFNet, self).__init__()
        self.channels = channels
        self.final_module = final_module
        self.in_channel = in_channel
        self.activation_type = activation_type
        self.activation_layer = Activation_Layer(activation_type)
        self.cell_out_multiple = channels[final_module[-1].toIndex]
        self.cell_1 = Cell(64,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.cell_2 = Cell(128,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.cell_3 = Cell(256,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.filter = tf.ones((3, 3, self.in_channel, 64), dtype=tf.float32)
        self.depthwise_filter_1 = tf.ones((3, 3, 64*self.cell_out_multiple, 1), tf.float32)
        self.pointwise_filter_1 = tf.constant(value=1.0, shape=[1, 1, 64*self.cell_out_multiple, 128],
                                       dtype=tf.float32)
        self.depthwise_filter_2 = tf.ones((3, 3, 128*self.cell_out_multiple, 1), tf.float32)
        self.pointwise_filter_2 = tf.constant(value=1.0, shape=[1, 1, 128*self.cell_out_multiple, 256],
                                       dtype=tf.float32)
        self.depthwise_filter_3 = tf.ones((3, 3, 256*self.cell_out_multiple, 1), tf.float32)
        self.pointwise_filter_3 = tf.constant(value=1.0, shape=[1, 1, 256*self.cell_out_multiple, 256],
                                       dtype=tf.float32)
    def call(self,inputs):
        out = tf.nn.conv2d(input = inputs, filters = self.filter, strides=[1, 1, 1, 1], padding='SAME')
        thismean, thisvariance = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean=thismean, variance=thisvariance,
                                               offset=None, scale=None, variance_epsilon=1e-5)
        out = self.activation_layer(out)

        out = self.cell_1(out)
        #注：tensorflow的same模式是往右下填充0，而不是包了一层。
        out = tf.nn.separable_conv2d(input = out, depthwise_filter = self.depthwise_filter_1,
                                            pointwise_filter = self.pointwise_filter_1,
                                            strides=[1, 2, 2, 1], padding='SAME')
        thismean, thisvariance = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean=thismean, variance=thisvariance,
                                        offset=None, scale=None, variance_epsilon=1e-5)
        out = self.activation_layer(out)

        out = self.cell_2(out)

        out = tf.nn.separable_conv2d(input = out, depthwise_filter = self.depthwise_filter_2,
                                            pointwise_filter = self.pointwise_filter_2,
                                            strides=[1, 2, 2, 1], padding='SAME')
        thismean, thisvariance = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean=thismean, variance=thisvariance,
                                        offset=None, scale=None, variance_epsilon=1e-5)
        out = self.activation_layer(out)


        out = self.cell_3(out)

        out = tf.nn.separable_conv2d(input = out, depthwise_filter = self.depthwise_filter_3,
                                            pointwise_filter = self.pointwise_filter_3,
                                            strides=[1, 1, 1, 1], padding='SAME')
        thismean, thisvariance = tf.nn.moments(out, axes=[0, 1, 2], keepdims=True)
        out = tf.nn.batch_normalization(out, mean=thismean, variance=thisvariance,
                                        offset=None, scale=None, variance_epsilon=1e-5)
        out = self.activation_layer(out)
        return out
