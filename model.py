import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import RevIN1d


# ==========================================
# 核心模块 1：时间均值注意力机制 (Temporal Mean Attention, TMA)
# 作用：作为自适应低通滤波器，动态评估局部时间窗口的全局能量分布，
#       有效平滑现实工业场景中的高频尖峰噪声。
# ==========================================
class TMeanSEBlock1d(nn.Module):
    def __init__(self, channels, reduction=4):
        super(TMeanSEBlock1d, self).__init__()
        # 确保降维后至少有 1 个维度
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()
        # 👑 全局均值池化 (剥离时间轴上的局部高频振荡，提取纯粹的通道能量描述符)
        y = x.mean(dim=2)  # [Batch, Channels]

        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1)
        # 动态通道门控加权：自适应抑制高频噪声通道，同时放行包含低频异常演化的关键通道
        return x * y


# ==========================================
# 核心模块 2：残差时空解耦块 (Residual Spatiotemporal Decoupling, Res-SD Block)
# 作用：严格隔离时间演化与空间拓扑特征的提取过程，防止多元物理量发生特征耦合；
#       并通过残差连接防止梯度消失，保留原始信号的低频基线信息。
# ==========================================
class ResSDBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super(ResSDBlock, self).__init__()
        # Step A: 物理隔离 (Depthwise)
        # 强制卷积核仅在单一通道的时间轴上滑动，切断异构物理量间的交叉干扰
        self.depthwise = nn.Conv1d(in_c, in_c, kernel_size=k, stride=1, padding=k // 2, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)

        # Step B: 拓扑融合 (Pointwise)
        # 执行跨通道线性组合，重建多变量传感器网络的瞬态空间拓扑信息
        self.pointwise = nn.Conv1d(in_c, out_c, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        # Step C: 时间均值注意力 (TMA 降噪模块)
        self.se = TMeanSEBlock1d(out_c)

        # 👑 残差短接连接 (Shortcut)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            # 如果通道数发生变化，用 1x1 卷积对齐维度，确保稳定传输
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        # 1. 留存原始信号备份 (用于后续保留低频基线)
        residual = self.shortcut(x)

        # 2. 时空解耦特征提取 (先时间演化，后空间拓扑)
        out = self.depthwise(x)
        out = F.relu(self.bn1(out), inplace=True)

        out = self.pointwise(out)
        out = self.bn2(out)

        # 3. 动态均值通道赋权 (自适应抑制高频尖峰噪声)
        out = self.se(out)

        # 4. 残差相加 (将提纯后的异常特征与原始低频基线融合)
        out += residual
        return F.relu(out, inplace=True)



''''# ==========================================
# 消融实验版本：标准残差块 (无时空解耦 w/o SD)
# 作用：将时间与空间重新混合，用于证明“解耦”机制在缓解多变量通道干扰方面的核心价值
# ==========================================
class ResSDBlock(nn.Module):
    def __init__(self, in_c, out_c, k):
        super(ResSDBlock, self).__init__()

        # 👑 消融修改：砍掉 Depthwise 和 Pointwise 的物理隔离机制！
        # 换回标准的 1D 卷积，使不同物理含义的通道在提取时间特征时发生强制耦合
        self.standard_conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=1, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_c)

        # 保持均值通道注意力的初始化（在 w/o TMA 实验中会被注释掉前向调用）
        self.se = TMeanSEBlock1d(out_c)

        # 👑 保持残差短接连接 (Shortcut) 不变
        # 确保性能差异完全来源于“解耦卷积”，排除网络深度变化带来的干扰
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        # 1. 留存原始信号备份
        residual = self.shortcut(x)

        # 2. 传统特征提纯 (未解耦，时间和通道维度联合提取)
        out = self.standard_conv(x)
        out = self.bn(out)

        # 3. 动态均值赋权（在单纯验证解耦作用的实验中，保持注释状态以控制变量）
        # out = self.se(out)

        # 4. 残差相加
        out += residual
        return F.relu(out, inplace=True)
'''
class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, projection_dim=256, layers=[128, 256, 128, 64],
                 kss=[7, 5, 3, 3],
                 use_revin: bool = True,
                 revin_affine: bool = False,
                 revin_eps: float = 1e-5,
                 revin_min_sigma: float = 1e-5
                 ):
        super(PatchEncoder, self).__init__()
        self.layers = layers
        self.kss = kss
        self.projection_dim = projection_dim

        # ====================================================
        # ⚠️ 注意：由于前端引入了运动学扩展 (Kinematic Expansion)
        # 如果输入是单变量 (in_channels=1)，它将被动态扩展为原始值、速度、加速度三个物理维度
        # 因此第一层的实际输入通道数需要判定为 3
        # ====================================================
        self.actual_in_channels = 3 if in_channels == 1 else in_channels
        #self.actual_in_channels = in_channels

        # 0. 基础抗漂移模块 (使用实际通道数初始化)
        self.revin = None
        if use_revin:
            self.revin = RevIN1d(num_channels=self.actual_in_channels,
                                 eps=revin_eps,
                                 min_sigma=revin_min_sigma,
                                 affine=revin_affine)

        # 1. 构建残差时空解耦网络
        blocks = []
        for i in range(len(self.layers)):
            # 第一层的输入通道数使用 actual_in_channels
            in_c = layers[i - 1] if i > 0 else self.actual_in_channels
            out_c = self.layers[i]
            k = self.kss[i]
            # 直接调用残差解耦块
            blocks.append(ResSDBlock(in_c, out_c, k))

        self.convblocks = nn.ModuleList(blocks)

        # 2. 后端预测头 (保持接口严格一致)
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(self.layers[-1], self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim)
        )
        self.classification_head = nn.Linear(self.layers[-1] * 2, 1)

    def forward(self, x, return_embedding=False, return_projection=False):

        # ====================================================
        # 👑 核心创新点：单变量运动学扩展 (Kinematic Expansion)
        # 放置在网络最前端，通过计算离散一阶速度和二阶加速度，
        # 将 1 维孤立序列扩展为 3 维动态状态空间，有效缓解单变量信息匮乏问题
        # ====================================================
        if x.size(1) == 1:
            v = torch.diff(x, dim=-1, prepend=x[:, :, :1])
            a = torch.diff(v, dim=-1, prepend=v[:, :, :1])
            x = torch.cat([x, v, a], dim=1)

        # ====================================================
        # 下游的常规流转：此时单变量已被补全为 C=3 的动态特征，或本身即为多变量输入
        # ====================================================

        # 1. 实例级归一化 (RevIN 此时会对所有物理通道分别做归一化，消除分布漂移)
        if self.revin is not None:
            x = self.revin.norm(x)

        # 2. 依次穿过残差时空解耦网络
        for block in self.convblocks:
            x = block(x)

        # 3. 全局池化压缩为干净的一维表示 (1D clean embedding)
        h = self.fc_embedding(x).flatten(start_dim=1)

        if return_embedding:
            return h
        if return_projection:
            return self.projection_head(h)

        raise ValueError("The forward method is not designed to handle classification directly.")

    def embedding(self, x):
        return self.forward(x, return_embedding=True)

    def projection(self, h):
        return self.projection_head(h)