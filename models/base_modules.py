# 导入PyTorch核心库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入类型提示相关模块
from typing import Optional, List, Tuple


# 通道混洗函数，用于在分组卷积后重新排列通道顺序
# 这种操作可以增强不同组之间的信息交流
def channel_shuffle(x, groups):
    '''通道混洗操作
    通道混洗过程: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
    其中N是批次大小，C是通道数，H/W是高度/宽度，g是分组数'''
    # 获取输入张量的维度信息
    batchsize, num_channels, depth, height, width = x.data.size()
    # 计算每组的通道数
    channels_per_group = num_channels // groups
    # 重塑张量形状: [N, C, D, H, W] -> [N, g, C/g, D, H, W]
    x = x.view(batchsize, groups,
               channels_per_group, depth, height, width)
    # 转置操作: 交换group维度和channel_per_group维度
    # contiguous()确保内存连续，便于后续操作
    x = x.permute(0, 2, 1, 3, 4, 5).contiguous()   
    # 恢复原始形状: [N, C/g, g, D, H, W] -> [N, C, D, H, W]
    x = x.view(batchsize, num_channels, depth, height, width)
    return x


# MobileOneBlock类：实现MobileOne网络的基本构建块
# 支持训练模式和推理模式的切换，具有重参数化能力
class MobileOneBlock(nn.Module): #由 Apple 团队在 2022 年提出的一种专为移动端（Mobile Devices）设计的轻量级神经网络架构
    # 初始化MobileOneBlock
    # 参数说明：
    # in_channels: 输入通道数
    # out_channels: 输出通道数
    # kernel_size: 卷积核大小
    # stride: 步长，默认(1,1,1)
    # padding: 填充大小
    # dilation: 膨胀率
    # groups: 分组卷积的组数
    # inference_mode: 是否为推理模式
    # use_se: 是否使用SE注意力机制
    # num_conv_branches: 卷积分支数量
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Tuple = (1, 1, 1),
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        # 调用父类初始化方法
        super(MobileOneBlock, self).__init__()
        # 存储各种配置参数
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # SE注意力模块（这里使用恒等映射作为占位符）没有真正实现SE注意力
        # SE 注意力模块通常指 Squeeze-and-Excitation（SE），是学一个权重告诉网络“哪些通道更重要”，然后对通道进行按权重的缩放。
        self.se = nn.Identity()
        # ReLU激活函数
        self.activation = nn.ReLU()

        # 根据模式选择不同的网络结构
        if inference_mode:
            # 推理模式：使用重参数化后的单一卷积层
            self.reparam_conv = nn.Conv3d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # 训练模式：定义残差连接（当输入输出通道数相同且步长为1时）
            self.rbr_skip = nn.BatchNorm3d(num_features=in_channels) \
                if out_channels == in_channels and stride == (1, 1, 1) else None

            # 创建多个并行的卷积分支
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                # 为每个分支创建卷积+BN组合
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            # 使用ModuleList管理多个分支
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # 1x1卷积分支（用于尺度变换）
            self.rbr_scale = None
            if kernel_size > 1:
                # 只有当kernel_size大于1时才需要1x1分支
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # def 函数名(参数名: 参数类型) -> 返回值类型:
        # 根据模式选择不同的前向传播路径
        if self.inference_mode:
            # 推理模式：直接通过重参数化的卷积层
            return self.activation(self.se(self.reparam_conv(x)))

        # 训练模式：计算各个分支的输出
        identity_out = 0
        if self.rbr_skip is not None:
            # 残差分支输出
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            # 1x1卷积分支输出
            scale_out = self.rbr_scale(x)

        # 融合所有分支的输出
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            # 累加所有并行卷积分支的输出
            out += self.rbr_conv[ix](x)

        # 应用SE注意力和激活函数后返回
        return self.activation(self.se(out))

    # 重参数化方法：将训练时的多分支结构合并为推理时的单一分支
    def reparameterize(self):
        # 如果已经是推理模式则直接返回
        if self.inference_mode:
            return
        # 获取融合后的卷积核和偏置
        kernel, bias = self._get_kernel_bias()
        # 创建重参数化后的卷积层
        self.reparam_conv = nn.Conv3d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        # 设置重参数化卷积层的参数
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 清理训练时的参数和模块
        for para in self.parameters():
            para.detach_()  # 分离参数梯度图
        # 删除训练时的分支模块
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        # 切换到推理模式
        self.inference_mode = True

    # 获取融合后的卷积核和偏置参数
    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 处理1x1卷积分支
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            # 融合1x1卷积分支的BN参数
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # 对1x1卷积核进行填充以匹配主卷积核大小
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad, pad, pad])

        # 处理恒等映射分支
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            # 融合恒等映射分支的BN参数
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # 处理并行卷积分支
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            # 融合每个卷积分支的BN参数
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        # 将所有分支的参数相加得到最终参数
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    # 融合卷积层和BN层的参数
    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        # 处理Sequential类型的分支（卷积+BN组合）
        if isinstance(branch, nn.Sequential):
            # 提取卷积层和BN层的参数
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # 处理纯BN层（恒等映射分支）
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                # 构造恒等映射的卷积核
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                # 在对角线上设置1，其余位置为0
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2,
                                    self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            # 使用构造的恒等卷积核
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        # 执行BN参数融合计算
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        # 返回融合后的卷积核和偏置
        return kernel * t, beta - running_mean * gamma / std

    # 创建卷积+BN的组合模块
    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        # 创建Sequential容器
        mod_list = nn.Sequential()
        # 添加3D卷积层
        mod_list.add_module('conv', nn.Conv3d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        # 添加批归一化层
        mod_list.add_module('bn', nn.BatchNorm3d(num_features=self.out_channels))
        return mod_list


# MobileOneBlocklist类：MobileOneBlock的列表组合模块
# 实现了经典的卷积块结构：1x1卷积 -> 深度可分离卷积 -> 1x1卷积
class MobileOneBlocklist(nn.Module):
    # 初始化MobileOneBlocklist
    # 参数说明：
    # in_planes: 输入通道数
    # out_planes: 输出通道数
    # num_blocks: 块的数量（未使用）
    # stride: 步长
    # use_se: 是否使用SE注意力
    # inference_mode: 推理模式标志
    # num_conv_branches: 卷积分支数
    # num_se_blocks: SE块数量（未使用）
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 num_blocks: int = 1,
                 stride: Tuple = (1, 1, 1),
                 use_se: bool = False,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1,
                 num_se_blocks: int = 0):
        # 调用父类初始化方法
        super(MobileOneBlocklist, self).__init__()

        # 第一个1x1逐点卷积：用于通道变换
        self.GConv1 = MobileOneBlock(in_channels=in_planes,
                                     out_channels=in_planes,
                                     kernel_size=1,
                                     stride=(1, 1, 1),
                                     padding=0,
                                     groups=1,
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)
        # 深度可分离卷积：逐通道卷积，'Wise'表示逐通道地处理
        self.DWConv = MobileOneBlock(in_channels=in_planes,
                                     out_channels=in_planes,
                                     kernel_size=3,
                                     stride=stride,
                                     padding=1,
                                     groups=in_planes,  # 分组数等于输入通道数，实现深度卷积
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)
        # 第二个1x1逐点卷积：用于输出通道变换
        self.GConv2 = MobileOneBlock(in_channels=in_planes,
                                     out_channels=out_planes,
                                     kernel_size=1,
                                     stride=(1, 1, 1),
                                     padding=0,
                                     groups=1,
                                     inference_mode=inference_mode,
                                     use_se=use_se,
                                     num_conv_branches=num_conv_branches)

    # 前向传播方法
    def forward(self, input):
        # 执行第一个1x1卷积
        out = self.GConv1(input)
        # 通道混洗：增强通道间的信息交流
        out = channel_shuffle(out, 2)
        # 执行深度可分离卷积
        out = self.DWConv(out)
        # 执行第二个1x1卷积
        out = self.GConv2(out)
        return out