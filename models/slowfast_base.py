# 导入PyTorch深度学习框架
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 从基础模块中导入MobileOneBlocklist组件
from models.base_modules import MobileOneBlocklist


# 定义基础3D卷积块类，继承自nn.Module
# 用于替换深度可分离卷积(Depthwise_Separable)的基础3D卷积模块
class BasicConv3D(nn.Module):
    """Basic 3D convolution block for replacing Depthwise_Separable"""

    # 初始化方法，定义3D卷积块的基本结构
    # 参数说明：
    # in_ch: 输入通道数
    # out_ch: 输出通道数
    # kernel_size: 卷积核大小，默认为1
    # stride: 步长，默认为1
    # padding: 填充大小，默认为0
    # bias: 是否使用偏置，默认为False
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False):
        # 调用父类nn.Module的初始化方法
        super(BasicConv3D, self).__init__()
        # 定义3D卷积层
        self.conv = nn.Conv3d(
            in_channels=in_ch,      # 输入通道数
            out_channels=out_ch,    # 输出通道数
            kernel_size=kernel_size, # 卷积核尺寸
            stride=stride,          # 卷积步长
            padding=padding,        # 填充大小
            bias=bias               # 是否使用偏置
        )
        # 批量归一化层，对3D数据进行归一化处理
        self.bn = nn.BatchNorm3d(out_ch)
        # ReLU激活函数，inplace=True表示直接修改输入数据以节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # 执行3D卷积操作
        out = self.conv(input)
        # 批量归一化处理
        out = self.bn(out)
        # ReLU激活函数
        out = self.relu(out)
        # 返回处理后的输出
        return out


# 定义基础侧向连接块类，继承自nn.Module
# 用于实现快慢路径之间的特征融合
class BasicLateral(nn.Module):
    """Basic lateral connection block"""

    # 初始化方法，定义侧向连接块的结构
    # 参数说明：
    # channels: 输入通道数
    def __init__(self, channels):
        # 调用父类nn.Module的初始化方法
        super(BasicLateral, self).__init__()
        # 定义3D卷积层用于侧向连接
        self.conv = nn.Conv3d(
            in_channels=channels,       # 输入通道数
            out_channels=channels * 2,  # 输出通道数为输入的2倍
            kernel_size=(5, 1, 1),      # 卷积核尺寸：时间维度5，空间维度1×1
            stride=(4, 1, 1),           # 时间维度步长4，空间维度步长1
            padding=(2, 0, 0),          # 时间维度填充2，空间维度不填充
            bias=False                  # 不使用偏置
        )
        # 对输出通道进行批量归一化
        self.bn = nn.BatchNorm3d(channels * 2)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

    # 前向传播方法
    def forward(self, x):
        # 执行侧向连接的卷积操作
        out = self.conv(x)
        # 批量归一化
        out = self.bn(out)
        # ReLU激活
        out = self.relu(out)
        # 返回处理结果
        return out


# 默认参数配置字典
# 包含模型训练和架构的各种超参数
default_params = {
    'width_multipliers': (0.5, 0.5, 0.5, 0.5),  # 宽度乘数，用于控制各层通道数的比例，实现轻量化设计，减少参数量和计算量
    'num_feature': 6,           # 特征数量
    'lateral': False,           # 是否启用侧向连接
    'num_classes': 12,          # 分类模型的总输出类别数
    'actions_classes': 6,       # 动作分类的类别数量
    'users_classes': 15,        # 用户身份分类的类别数量
    'epoch_num': 150,           # 训练轮数
    'batch_size': 16,           # 批次大小
    'learning_rate': 0.01,      # 学习率
}


# SlowFast双路径网络主类，继承自nn.Module
class SlowFast(nn.Module):
    # 初始化SlowFast网络
    # 参数说明：
    # actclass_num: 动作分类类别数，默认使用默认参数
    # userclass_num: 用户分类类别数，默认使用默认参数
    # dropout: Dropout比率，默认0.5
    # width_multipliers: 宽度乘数元组，默认使用默认参数
    # C_multipliers: 通道乘数元组，默认(0.5,)
    def __init__(self, actclass_num=default_params['actions_classes'],
                 userclass_num=default_params['users_classes'],
                 dropout=0.5,
                 width_multipliers=default_params['width_multipliers'],
                 C_multipliers=(0.5,)):
        # 调用父类初始化方法
        super(SlowFast, self).__init__()

        # 快路径定义部分
        # 快路径第一层卷积：处理高频信息，时间分辨率高
        self.fast_conv1 = BasicConv3D(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        # 快路径最大池化层
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 快路径主要构建块
        # 第二阶段块：输入8通道，输出32*width_multipliers[0]通道
        self.fast_shu2_1 = MobileOneBlocklist(in_planes=8, out_planes=int(32 * width_multipliers[0]), stride=(1, 2, 2))
        # 第三阶段块：输入32*width_multipliers[0]通道，输出64*width_multipliers[1]通道
        self.fast_shu3_1 = MobileOneBlocklist(int(32 * width_multipliers[0]), out_planes=int(64 * width_multipliers[1]),
                                              stride=(1, 2, 2))
        # 第四阶段块：输入64*width_multipliers[1]通道，输出128*width_multipliers[2]通道
        self.fast_shu4_1 = MobileOneBlocklist(in_planes=int(64 * width_multipliers[1]),
                                              out_planes=int(128 * width_multipliers[2]), stride=(1, 2, 2))
        # 第五阶段块：输入128*width_multipliers[2]通道，输出256*width_multipliers[3]通道
        self.fast_shu5 = MobileOneBlocklist(in_planes=int(128 * width_multipliers[2]),
                                            out_planes=int(256 * width_multipliers[3]))

        # 侧向连接定义（基础实现版本）
        # 第一层侧向连接：处理8通道特征
        self.lateral_p1 = BasicLateral(8)
        # 第二层侧向连接：处理32*width_multipliers[0]通道特征
        self.lateral_res2 = BasicLateral(int(32 * width_multipliers[0]))
        # 第三层侧向连接：处理64*width_multipliers[1]通道特征
        self.lateral_res3 = BasicLateral(int(64 * width_multipliers[1]))
        # 第四层侧向连接：处理128*width_multipliers[2]通道特征
        self.lateral_res4 = BasicLateral(int(128 * width_multipliers[2]))

        # 慢路径定义部分
        # 慢路径第一层卷积：处理低频信息，时间分辨率较低
        self.slow_conv1 = BasicConv3D(3, int(64 * C_multipliers[0]), kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)
        # 慢路径最大池化层
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 慢路径主要构建块
        # 第二阶段块：输入通道包括侧向连接特征(16)和原始特征(64*C_multipliers[0])
        self.slow_shu2_1 = MobileOneBlocklist(in_planes=int(16 + (64 * C_multipliers[0])),
                                              out_planes=int(256 * width_multipliers[0] * C_multipliers[0]),
                                              stride=(1, 2, 2))
        # 第三阶段块：融合快路径特征和慢路径特征
        self.slow_shu3_1 = MobileOneBlocklist(
            in_planes=int((64 * width_multipliers[0]) + 256 * width_multipliers[0] * C_multipliers[0]),
            out_planes=int(512 * width_multipliers[1] * C_multipliers[0]),
            stride=(1, 2, 2))
        # 第四阶段块：继续特征融合
        self.slow_shu4_1 = MobileOneBlocklist(
            in_planes=int((128 * width_multipliers[1]) + 512 * width_multipliers[0] * C_multipliers[0]),
            out_planes=int(1024 * width_multipliers[2] * C_multipliers[0]),
            stride=(1, 2, 2))
        # 第五阶段块：最终特征提取
        self.slow_shu5 = MobileOneBlocklist(
            in_planes=int((256 * width_multipliers[2]) + 1024 * width_multipliers[2] * C_multipliers[0]),
            out_planes=int(2048 * width_multipliers[3] * C_multipliers[0]))

        # 输出层定义
        # Dropout层用于防止过拟合
        self.dp = nn.Dropout(dropout)
        # 动作分类全连接层：输入为快慢路径特征拼接，输出动作类别
        self.fcx = nn.Linear(int(256 * width_multipliers[3] + (2048 * width_multipliers[3] * C_multipliers[0])),
                             actclass_num, bias=True)
        # 用户分类全连接层：输入相同，输出用户类别
        self.fcu = nn.Linear(int(256 * width_multipliers[3] + (2048 * width_multipliers[3] * C_multipliers[0])),
                             userclass_num, bias=True)

    def forward(self, input):
        # 处理快路径：时间步长为4，提取高频信息
        fast, lateral = self.FastPath(input[:, :, ::4, :, :])  # temporal stride 4
        # 处理慢路径：时间步长为16，提取低频信息，并接收来自快路径的侧向连接
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral)  # temporal stride 16
        # 合并快慢路径特征：在通道维度上拼接
        x = torch.cat([slow, fast], dim=1)

        # 输出头部处理
        a = self.dp(x)      # Dropout正则化
        a = self.fcx(a)     # 动作分类预测

        return a            # 返回动作分类结果

    # 慢路径处理方法
    def SlowPath(self, input, lateral):
        # 慢路径第一层卷积和池化
        x = self.slow_conv1(input)
        x = self.slow_maxpool(x)
        # 第一次特征融合：拼接来自快路径的第一层侧向连接特征
        x = torch.cat([x, lateral[0]], dim=1)

        # 第二阶段处理和特征融合
        x = self.slow_shu2_1(x)
        # 第二次特征融合：拼接第二层侧向连接特征
        x = torch.cat([x, lateral[1]], dim=1)

        # 第三阶段处理和特征融合
        x = self.slow_shu3_1(x)
        # 第三次特征融合：拼接第三层侧向连接特征
        x = torch.cat([x, lateral[2]], dim=1)

        # 第四阶段处理和特征融合
        x = self.slow_shu4_1(x)
        # 第四次特征融合：拼接第四层侧向连接特征
        x = torch.cat([x, lateral[3]], dim=1)

        # 第五阶段处理
        x = self.slow_shu5(x)
        # 自适应平均池化：将空间维度压缩为1×1×1
        x = nn.AdaptiveAvgPool3d(1)(x)
        # 展平操作：转换为(batch_size, channels)格式
        x = x.view(-1, x.size(1))
        return x

    # 快路径处理方法
    def FastPath(self, input):
        # 初始化侧向连接特征列表
        lateral = []

        # 快路径第一层处理
        x = self.fast_conv1(input)
        pool1 = self.fast_maxpool(x)  # 最大池化

        # 第一层侧向连接生成
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)  # 第一次融合，保存供慢路径使用

        # 第二阶段处理和侧向连接
        shuf2_1 = self.fast_shu2_1(pool1)
        lateral_res2 = self.lateral_res2(shuf2_1)
        lateral.append(lateral_res2)  # 第二次融合，保存供慢路径使用

        # 第三阶段处理和侧向连接
        shuf3_1 = self.fast_shu3_1(shuf2_1)
        lateral_res3 = self.lateral_res3(shuf3_1)
        lateral.append(lateral_res3)  # 第三次融合，保存供慢路径使用

        # 第四阶段处理和侧向连接
        shuf4_1 = self.fast_shu4_1(shuf3_1)
        lateral_res4 = self.lateral_res4(shuf4_1)
        lateral.append(lateral_res4)  # 第四次融合，保存供慢路径使用

        # 第五阶段处理
        shuf5 = self.fast_shu5(shuf4_1)
        # 自适应平均池化
        x = nn.AdaptiveAvgPool3d(1)(shuf5)
        # 展平操作
        x = x.view(-1, x.size(1))

        # 返回快路径特征和所有侧向连接特征
        return x, lateral