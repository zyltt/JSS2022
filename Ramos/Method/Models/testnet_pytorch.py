import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation_Layer(nn.Module):
    def __init__(self,type):
        super(Activation_Layer, self).__init__()
        self.type = type
    def forward(self,x):
        if self.type == "relu":
            return F.relu(x)
        elif self.type == "sigmoid":
            return torch.sigmoid(x)
        elif self.type == "tanh":
            return torch.tanh(x)
        elif self.type == "leakyrelu":
            return torch.nn.LeakyReLU(negative_slope=0.2)(x)
        elif self.type == "prelu":
            #mindspore在cpu下不支持prelu，因此就不进行计算了
            return x
        elif self.type == "elu":
            return torch.nn.ELU()(x)


class Cell(nn.Module):
    def __init__(self,in_channel,final_module,channels,activation_type):
        super(Cell, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
        self.activation_layer = Activation_Layer(activation_type)
    def forward(self,x):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(x)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            fromIndex = eachOperation.fromIndex
            final_point = eachOperation.toIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # 本NAS规定所有操作出通道数与入通道数相同。
            operator_in_channel = self.channels[fromIndex]*self.in_channel
            operator = eachOperation.operator
            # indentity
            if operator == -1:
                # print("pytorch执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = tensors[fromIndex].clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = torch.cat((tensors[toIndex], input), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("pytorch执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是1*1卷积的代码。
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 1, 1]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("pytorch执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    # 注：pytorch中dw卷积加pw卷积是普通卷积操作加groups来调节的。
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(operator_in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(operator_in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 3:
                # print("pytorch执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    depthwise_filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    pointwise_filter = torch.ones((self.in_channel, operator_in_channel, 1, 1),
                                                  dtype=torch.float32)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    # 归一化和relu
                    pointwise_temp = torch.nn.BatchNorm2d(self.in_channel)(pointwise_temp)
                    pointwise_temp = self.activation_layer(pointwise_temp)
                    tensors[toIndex] = pointwise_temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    depthwise_filter = torch.ones((operator_in_channel, 1, 3, 3), dtype=torch.float32)
                    pointwise_filter = torch.ones((self.in_channel, operator_in_channel, 1, 1),
                                                  dtype=torch.float32)
                    pad = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    # 归一化和relu
                    pointwise_temp = torch.nn.BatchNorm2d(self.in_channel)(pointwise_temp)
                    pointwise_temp = self.activation_layer(pointwise_temp)
                    tensors[toIndex] = torch.cat((tensors[toIndex], pointwise_temp), 1).clone().detach()
            elif operator == 4:
                # print("pytorch执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 5:
                # print("pytorch执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                    result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True,
                                                count_include_pad=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True,
                                                count_include_pad=False)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("pytorch执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 3, 3]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    # 归一化和relu

                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = torch.Tensor(np.ones([self.in_channel, operator_in_channel, 3, 3]))
                    thisresult = F.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator == 7:
                # print("pytorch执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = torch.Tensor(np.ones([operator_in_channel, self.in_channel, 3, 3]))
                    thisresult = F.conv_transpose2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    # 归一化和relu

                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = torch.Tensor(np.ones([operator_in_channel, self.in_channel, 3, 3]))
                    thisresult = F.conv_transpose2d(input=input, weight=filter, stride=[1, 1], padding=1)
                    # 归一化和relu
                    thisresult = torch.nn.BatchNorm2d(self.in_channel)(thisresult)
                    thisresult = self.activation_layer(thisresult)
                    tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #relu
            elif operator == 8:
                # print("pytorch执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = F.relu(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = F.relu(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #sigmoid
            elif operator == 9:
                # print("pytorch执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.sigmoid(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.sigmoid(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #tanh
            elif operator == 10:
                # print("pytorch执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.tanh(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.tanh(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #leakyrelu
            elif operator == 11:
                # print("pytorch执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.LeakyReLU(negative_slope=0.2)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.LeakyReLU(negative_slope=0.2)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #prelu。注：mindspore cpu不支持prelu，故直接不用即可。
            elif operator == 12:
                # print("pytorch执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = input
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = input
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #ELU
            elif operator == 13:
                # print("pytorch执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.ELU()(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.ELU()(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #batchnorm
            elif operator == 14:
                # print("pytorch执行了了操作14 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = torch.nn.BatchNorm2d(operator_in_channel)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = torch.nn.BatchNorm2d(operator_in_channel)(input)
                    temp = torch.cat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
        return tensors[final_point].clone().detach()



class TorchNet(nn.Module):
    def __init__(self, channels, final_module, in_channel, activation_type):
        super(TorchNet, self).__init__()
        self.channels = channels
        self.final_module = final_module
        self.in_channel = in_channel
        self.activation_type = activation_type
        self.activation_layer = Activation_Layer(activation_type)
        self.cell_out_multiple = channels[final_module[-1].toIndex]
        self.cell_1 = Cell(64,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.cell_2 = Cell(128,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.cell_3 = Cell(256,final_module=self.final_module,channels=self.channels,activation_type=self.activation_type)
        self.filter = torch.Tensor(np.ones([64,self.in_channel,3,3]))
        self.depthwise_filter_1 = torch.ones((64*self.cell_out_multiple, 1, 3, 3), dtype=torch.float32)
        self.pointwise_filter_1 = torch.ones((128, 64*self.cell_out_multiple, 1, 1), dtype=torch.float32)
        self.depthwise_filter_2 = torch.ones((128*self.cell_out_multiple, 1, 3, 3), dtype=torch.float32)
        self.pointwise_filter_2 = torch.ones((256, 128*self.cell_out_multiple, 1, 1), dtype=torch.float32)
        self.depthwise_filter_3 = torch.ones((256*self.cell_out_multiple, 1, 3, 3), dtype=torch.float32)
        self.pointwise_filter_3 = torch.ones((256, 256*self.cell_out_multiple, 1, 1), dtype=torch.float32)
        #注：tensorflow的same模式是往右下填充0，而不是包了一层。为了保持一致，pytorch中应当单独写往右和下的pad
        #顺序：左右上下
        #pad_1用于减数的separable_conv
        self.pad_1 = torch.nn.ZeroPad2d(padding=(0, 2, 0, 2))
        #pad_2用于非减数的separable_conv
        #注：tensorflow中 步长为奇数时两端补0，步长为偶数时右下补0。因此需要单独写。
        self.pad_2 = torch.nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.avgpool = nn.AvgPool2d(8, stride=1)
    def forward(self,x):
        out = F.conv2d(input=x, weight=self.filter, stride=[1, 1], padding=1)
        out = torch.nn.BatchNorm2d(64)(out)
        out = self.activation_layer(out)

        out = self.cell_1(out)

        out = self.pad_1(out)
        #注：tensorflow中的separable conv中的stride是针对depthwise conv部分的
        out = F.conv2d(input=out, weight=self.depthwise_filter_1, stride=2,
                                                    padding=0, groups=64*self.cell_out_multiple)
        out = F.conv2d(input=out, weight=self.pointwise_filter_1, stride=1,
                                                    padding=0)
        out = torch.nn.BatchNorm2d(128)(out)
        out = self.activation_layer(out)

        out = self.cell_2(out)

        out = self.pad_1(out)
        out = F.conv2d(input=out, weight=self.depthwise_filter_2, stride=2,
                                                    padding=0, groups=128*self.cell_out_multiple)
        out = F.conv2d(input=out, weight=self.pointwise_filter_2, stride=1,
                                                    padding=0)
        out = torch.nn.BatchNorm2d(256)(out)
        out = self.activation_layer(out)


        out = self.cell_3(out)

        out = self.pad_2(out)
        out = F.conv2d(input=out, weight=self.depthwise_filter_3, stride=1,
                                                    padding=0, groups=256 * self.cell_out_multiple)
        out = F.conv2d(input=out, weight=self.pointwise_filter_3, stride=1,
                                                    padding=0)
        out = torch.nn.BatchNorm2d(256)(out)
        out = self.activation_layer(out)
        return out