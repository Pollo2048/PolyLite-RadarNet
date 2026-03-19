import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
from models.slowfast_base import SlowFast
from data.dataset import VideoDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):  #跟踪 loss、accuracy 等指标的平均值和当前值。
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()  # 初始化时重置所有参数

    def reset(self):
        self.val = 0   # 当前批次的指标值
        self.avg = 0     # 累计平均值
        self.sum = 0    # 累计总和
        self.count = 0  # 累计样本数/批次数量

    """
    更新统计变量
    Args:
        val: 当前批次的指标值（如当前batch的loss）
        n: 该批次的样本数量，默认1
    """
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n # 累加总指标值
        self.count += n # 累加样本数
        self.avg = self.sum / self.count    # 计算平均值


    """
    计算指定top-k精度（此处默认top-1）
    Args:
        output: 模型输出的预测概率/对数概率，shape=(batch_size, num_classes)
        target: 真实标签，shape=(batch_size,)
        topk: 要计算的top-k值，默认(1,)表示仅计算top-1精度 是一个元组tuple
    Returns:
        res: 包含top-k精度和预测标签的列表，格式[top1_acc, pred_labels]
    """
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)     # 取最大的k个值（此处为1）
    batch_size = target.size(0)  #获取 batch 中样本数量

    _, pred = output.topk(maxk, 1, True, True)  #output.topk(k, dim, largest, sorted)
    pred_labels = pred[:, 0] #取第一列 在PyTorch 中，张量的索引格式是 [行, 列]
    pred = pred.t()  #转置
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #expand时每个维度要么相等，要么原张 量该维度为 1
    #例 correct = [[ True, True, False, True]]
    res = []
    for k in topk:
        # .view(-1) → 展平为一维向量 该操作需要内存连续 所以需要contiguous float()把布尔转化为浮点 便于后续乘以 100.0 / batch_size
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size)) # 准确率 行业标准惯例用 75 而不是0.75或 3/4
    res.append(pred_labels) #记录预测标签
    return res


def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter() #每个 batch 的总处理时间（数据加载 + 前向 + 反向 + 优化）
    data_time = AverageMeter() #纯数据加载时间（从 dataloader 取出数据到送入 GPU 的耗时）
    losses = AverageMeter() #每个 batch 的损失值（loss.item()）
    top1 = AverageMeter() #每个 batch 的 Top-1 精度（来自 accuracy() 函数返回的 prec1）

    model.train() #将模型切换到“训练模式
    end = time.time() #记录当前时间戳（秒），用于后续计算耗时

    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end) #数据加载时间

        inputs = inputs.cuda() #把数据放到默认 GPU 设备上
        labels = labels.cuda()

        # Forward pass
        outputs = model(inputs)  #shape [batch_size, num_classes] 每个样本在每个类别上的得分
        loss = criterion(outputs, labels)  #计算模型预测结果和真实标签之间的差距 常用的是交叉熵损失

        # Measure accuracy and record loss
        prec1, pred_labels = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad() #所有可训练参数的梯度清零
        loss.backward() #当前 batch 的 loss 执行反向传播
        optimizer.step() #更新模型参数

        batch_time.update(time.time() - end) #batch 总耗时
        end = time.time()

        #打印每个epoch的损失和准确率
        if (step + 1) % 100 == 0:
            print(f'Epoch: [{epoch}][{step + 1}/{len(train_dataloader)}]')
            print(f'Loss {losses.avg:.4f}\t'
                  f'Acc@1 {top1.avg:.3f}')

    # Log to tensorboard
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc', top1.avg, epoch)

    return top1.avg, losses.avg


def validate(model, val_dataloader, epoch, criterion, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (inputs, labels) in enumerate(val_dataloader):
            # (index, (data, target))
            # enumerate() 是Python内置函数 为可迭代对象添加计数器 返回 (索引, 元素) 的元组序列
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Measure accuracy and record loss
            prec1, pred_labels = accuracy(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        #
        # print(f'Validation Results - Epoch: [{epoch}]')
        # print(f'Loss {losses.avg:.4f}\t'
        #       f'Acc@1 {top1.avg:.3f}')

    # Log to tensorboard
    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg


def main():
    # Set random seed for reproducibility
    torch.manual_seed(3407) #训练过程 多环节涉及“随机性  权重初始化 shuffle 随机打乱样本顺序  Dropout 层：随机丢弃神经元
    cudnn.benchmark = False #深度神经网络的加速库。针对同一种卷积操作（Convolution），cuDNN 提供了多种不同的计算算法  增强可复现性
    datadir = "./data/dataset"

    # Setup tensorboard  tensorboard --logdir 2025-12-04-14-25-08 数据可视化
    #time.time() 返回一个浮点数，表示Unix 纪元（1970年1月1日 00:00:00 UTC）以来经过的秒数
    #strftime 将结构化的时间对象按照指定的格式（Format）转换为字符串
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_dir = os.path.join('logs', cur_time)
    writer = SummaryWriter(log_dir=log_dir) #实例化后，它会在指定的路径下创建一个事件文件 后续代码中调
                                            # 用的 writer.add_scalar() 等方法都会通过这个 writer 对象将数据（如 Loss、Accuracy）写入到该目录中

    # Create model save directory
    model_save_dir = os.path.join('checkpoints', cur_time)
    os.makedirs(model_save_dir, exist_ok=True) #exist_ok=True：这是一个关键的安全参数 目标文件夹已经存在时，程序会报错并崩溃
                                              # True，则表示“如果文件夹已存在，请保持静默，不要报错，继续运行”

    # Dataset loading code here
    # Replace with your actual dataset and dataloader creation
    dataset = VideoDataset(directory=datadir, clip_len=64) #雷达数据存放的根目录  每个视频样本应包含 64 帧雷达信号数据
    train_size = int(0.7 * len(dataset)) #将总数的 70% 分给训练集
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) #PyTorch 工具函数，用于将一个数据集随机打乱并拆分为多个不重叠的子数据集
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True) #每个训练 Epoch 开始前再次打乱数据顺序 防止模型记住样本出现的固定顺序
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create model
    model = SlowFast()  # Replace with your model  实例化模型
    model = model.cuda()#将模型的所有参数（Weights）和缓冲区（Buffers）转移到 NVIDIA GPU 的显存

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda() # .cuda() 是搬运到 GPU
    # 优化器的作用是根据损失函数计算出的梯度（Gradient）来更新模型的权重参数
    optimizer = optim.SGD(model.parameters(),  #告诉优化器需要更新哪些参数。它会遍历 SlowFast 模型中所有卷积层、全连接层等具备权重的层
                          lr=0.1,  # Learning rate
                          momentum=0.9, #它会累积之前的梯度方向，帮助优化器在“坑洼”的损失平面上更快地滑向目标点，并减少震荡
                          weight_decay=1e-4) #权重衰减 它通过对较大的权重施加惩罚来防止模型过拟合

    # Learning rate scheduler 学习率调度器 是一种“阶梯式”衰减策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, #指定该调度器作用于哪一个优化器
                                          step_size=30,  # Decay steps 触发周期 每隔 30 个 Epoch，学习率就会进行一次更新
                                          gamma=0.1)  # Decay rate 衰减倍数：学习率更新时的缩放比例

    best_acc = 0
    for epoch in range(100):  # Number of epochs  主训练循环 (Main Training Loop)
        print(f"Epoch: {epoch}")

        # Train for one epoch
        train_acc, train_loss = train(model, train_dataloader, epoch,
                                      criterion, optimizer, writer)
        print("train_acc:",train_acc)
        print("train_loss:",train_loss)

        # Evaluate on validation set  验证阶段
        val_acc, val_loss = validate(model, val_dataloader, epoch,
                                     criterion, writer)

        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(model_save_dir, 'best_model.pth'))

        scheduler.step()

    writer.close()
    print(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()

#tensorboard --logdir 2025-12-04-14-25-08 数据可视化