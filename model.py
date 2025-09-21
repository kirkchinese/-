"""
DiffPhys-UAV神经网络控制器模型

这是一个基于视觉的循环神经网络控制器，用于无人机的自主导航和避障。
模型结合了卷积神经网络（处理深度图像）和门控循环单元（处理时序信息），
实现端到端的感知-决策-控制流程。

主要特性：
- 视觉感知：处理64x48深度图像输入
- 多模态融合：结合视觉特征和状态向量
- 时序建模：使用GRU处理时间序列信息
- 控制输出：生成3D加速度和速度预测

网络架构：
1. 卷积特征提取器：深度图像 → 视觉特征
2. 状态投影层：状态向量 → 状态特征  
3. 特征融合：视觉特征 + 状态特征
4. 时序建模：GRU循环神经网络
5. 控制输出：全连接层 → 控制动作

依赖：
    - torch: PyTorch深度学习框架
    - torch.nn: 神经网络模块
"""

import torch
from torch import nn

def g_decay(x, alpha):
    """
    梯度衰减函数
    
    这是一个用于控制梯度传播的辅助函数，通过线性插值的方式
    在可微分路径和截断梯度路径之间进行平衡。
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        alpha (float): 衰减因子，范围[0, 1]
                      alpha=1: 完全保留梯度
                      alpha=0: 完全截断梯度
                      
    Returns:
        torch.Tensor: 输出张量，与输入形状相同
                     输出 = alpha * x + (1-alpha) * x.detach()
                     
    Note:
        - x.detach()会截断梯度传播
        - 该函数允许在训练过程中动态调整梯度强度
        - 常用于稳定训练过程，防止梯度爆炸
    """
    return x * alpha + x.detach() * (1 - alpha)

class Model(nn.Module):
    """
    无人机视觉控制器神经网络模型
    
    这是一个多模态循环神经网络，结合了视觉感知和状态感知，
    用于无人机的端到端控制。网络架构包括卷积特征提取、
    状态嵌入、特征融合和时序建模。
    
    网络输入：
    - 视觉输入：64x48深度图像
    - 状态输入：包含目标速度、姿态、边距等信息的向量
    
    网络输出：
    - 控制动作：6维向量（3维加速度 + 3维速度预测）
    """
    
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        """
        初始化无人机控制器网络
        
        构建完整的神经网络架构，包括视觉特征提取器、状态投影层、
        循环神经网络和输出层。网络设计考虑了无人机控制的实时性要求。
        
        Args:
            dim_obs (int, optional): 状态观测向量维度. 默认: 9
                                    通常包含：
                                    - 目标速度（本体坐标系）: 3维
                                    - 当前姿态（上向量）: 3维  
                                    - 安全边距: 1维
                                    - 可选：当前速度（本体坐标系）: 3维
            dim_action (int, optional): 动作输出维度. 默认: 4
                                       实际使用中通常为6维：
                                       - 预测加速度: 3维
                                       - 预测速度: 3维
                                       
        Network Architecture:
            1. 视觉特征提取器 (stem)：
               - Conv2d(1→32): 64x48 → 32x12 (kernel=2, stride=2)
               - Conv2d(32→64): 32x12 → 64x4x6 (kernel=3)  
               - Conv2d(64→128): 64x4x6 → 128x2x4 (kernel=3)
               - Linear: 128*2*4 → 192
               
            2. 状态投影层 (v_proj)：
               - Linear: dim_obs → 192
               
            3. 时序建模 (gru)：
               - GRUCell: 192 → 192 (隐藏状态维度)
               
            4. 控制输出 (fc)：
               - Linear: 192 → dim_action
        """
        super().__init__()
        
        # ================ 视觉特征提取器 ================
        """
        卷积神经网络：从深度图像中提取空间特征
        输入：1通道深度图像 (batch, 1, 64, 48)
        输出：192维特征向量
        """
        self.stem = nn.Sequential(
            # 第一层：下采样和初步特征提取
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 64x48 → 32x24 → 16x12 (stride=2)
            nn.LeakyReLU(0.05),
            
            # 第二层：空间特征聚合
            nn.Conv2d(32, 64, 3, bias=False),    # 16x12 → 14x10 → 64x4x6
            nn.LeakyReLU(0.05),
            
            # 第三层：高级特征提取
            nn.Conv2d(64, 128, 3, bias=False),   # 14x10 → 12x8 → 128x2x4  
            nn.LeakyReLU(0.05),
            
            # 展平和全连接
            nn.Flatten(),                        # 128x2x4 → 1024
            nn.Linear(128*2*4, 192, bias=False), # 1024 → 192
        )
        
        # ================ 状态投影层 ================
        """
        将状态观测向量投影到与视觉特征相同的维度空间
        便于后续的多模态特征融合
        """
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)  # 权重初始化：减小初始影响

        # ================ 时序建模层 ================
        """
        门控循环单元：处理时间序列信息，记忆历史状态
        对于控制任务至关重要，能够处理部分观测和动态环境
        """
        self.gru = nn.GRUCell(192, 192)
        
        # ================ 控制输出层 ================
        """
        将循环网络的隐藏状态映射为具体的控制动作
        包括加速度指令和速度预测
        """
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)  # 权重初始化：减小初始动作幅度
        
        # ================ 激活函数 ================
        self.act = nn.LeakyReLU(0.05)  # 轻微负斜率，避免ReLU死亡问题

    def reset(self):
        """
        重置模型状态
        
        重置循环神经网络的隐藏状态，通常在每个episode开始时调用。
        对于GRU，隐藏状态会在forward函数中自动处理，所以这里是空实现。
        
        Returns:
            None: 该方法为接口预留，当前无具体操作
            
        Note:
            如果需要显式管理GRU的隐藏状态，可以在这里初始化
            例如：self.hidden = None
        """
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        """
        前向传播函数
        
        执行完整的网络前向计算，包括视觉特征提取、状态特征投影、
        多模态特征融合、时序建模和控制输出生成。
        
        Args:
            x (torch.Tensor): 深度图像输入 [batch_size, 1, height, width]
                             通常为 [batch_size, 1, 64, 48]
                             像素值应已经过预处理和归一化
            v (torch.Tensor): 状态观测向量 [batch_size, dim_obs]
                             包含以下信息：
                             - 目标速度（本体坐标系）[3D]
                             - 当前姿态（上向量）[3D]
                             - 安全边距 [1D]
                             - 可选：当前速度（本体坐标系）[3D]
            hx (torch.Tensor, optional): GRU隐藏状态 [batch_size, 192]
                                        如果为None，GRU会自动初始化零状态
                                        
        Returns:
            tuple: (act, values, hx_new)
                - act (torch.Tensor): 控制动作输出 [batch_size, dim_action]
                                     通常包含：
                                     - 预测加速度指令 [3D]
                                     - 预测速度 [3D]
                - values (None): 价值函数输出（当前未实现，返回None）
                - hx_new (torch.Tensor): 更新后的GRU隐藏状态 [batch_size, 192]
                                        用于下一时间步的计算
                                        
        Forward Pass流程：
            1. 视觉特征提取：x → img_feat (192维)
            2. 状态特征投影：v → state_feat (192维)
            3. 特征融合：img_feat + state_feat → fused_feat
            4. 时序建模：fused_feat → GRU → hidden_state
            5. 控制输出：hidden_state → act
            
        Note:
            - 网络输出的动作在本体坐标系下，需要在使用时转换到世界坐标系
            - GRU的隐藏状态保持了历史信息，支持部分观测环境下的决策
            - 特征融合使用简单的加法，确保两种模态的平衡
        """
        # ================ 视觉特征提取 ================
        """
        通过卷积神经网络从深度图像中提取空间特征
        处理障碍物分布、距离信息等视觉线索
        """
        img_feat = self.stem(x)  # [batch, 1, 64, 48] → [batch, 192]
        
        # ================ 多模态特征融合 ================
        """
        将视觉特征和状态特征进行融合
        状态特征提供目标导向和运动状态信息
        """
        state_feat = self.v_proj(v)  # [batch, dim_obs] → [batch, 192]
        fused_feat = self.act(img_feat + state_feat)  # 特征加法融合 + 激活
        
        # ================ 时序建模 ================
        """
        使用GRU处理时间序列信息，整合历史观测
        对于动态环境下的决策至关重要
        """
        hx = self.gru(fused_feat, hx)  # [batch, 192], [batch, 192] → [batch, 192]
        
        # ================ 控制输出生成 ================
        """
        将时序特征映射为具体的控制动作
        输出包括加速度指令和速度预测
        """
        act = self.fc(self.act(hx))  # [batch, 192] → [batch, dim_action]
        
        return act, None, hx


# ================ 模型测试 ================
if __name__ == '__main__':
    """
    简单的模型实例化测试
    验证网络结构定义是否正确
    """
    model = Model()
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
