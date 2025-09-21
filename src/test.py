"""
test.py - 四旋翼无人机仿真CUDA扩展模块测试脚本

该脚本用于测试 quadsim_cuda 扩展模块的正确性，通过对比CUDA实现和PyTorch
纯Python实现的结果来验证前向传播和反向传播的准确性。

测试内容：
- 前向传播计算的准确性验证
- 反向传播梯度计算的准确性验证
- CUDA实现与PyTorch实现的数值一致性检查

主要功能：
- 生成随机测试数据
- 运行CUDA和PyTorch两种实现
- 对比计算结果并断言一致性
"""

import math
import torch
import quadsim_cuda


class GDecay(torch.autograd.Function):
    """
    梯度衰减自定义函数
    
    该类实现了一个自定义的PyTorch自动求导函数，用于在反向传播时
    对梯度进行衰减处理，模拟仿真中的数值阻尼效果。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向传播函数
        
        参数:
            ctx: PyTorch上下文对象，用于保存反向传播所需的信息
            x (torch.Tensor): 输入张量
            alpha (float): 衰减系数
            
        返回:
            torch.Tensor: 原样返回输入张量x（前向传播时不做任何变换）
        """
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数
        
        参数:
            ctx: PyTorch上下文对象，包含前向传播时保存的信息
            grad_output (torch.Tensor): 从上层传递下来的梯度
            
        返回:
            tuple: (对x的梯度, 对alpha的梯度)
                - 对x的梯度: grad_output * alpha (应用衰减)
                - 对alpha的梯度: None (alpha不需要梯度)
        """
        return grad_output * ctx.alpha, None

# 创建梯度衰减函数的可调用版本
g_decay = GDecay.apply

# ==================== 测试数据生成 ====================
# 生成64个四旋翼无人机的批量测试数据，所有数据都在CUDA设备上

# 旋转矩阵 (64批次, 3x3矩阵) - 描述无人机的姿态方向
R = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')

# 重力偏差向量 (64批次, 3维) - 模拟重力加速度的变化
dg = torch.randn((64, 3), dtype=torch.double, device='cuda')

# Z轴阻力系数 (64批次, 1维) - 垂直方向的空气阻力参数
z_drag_coef = torch.randn((64, 1), dtype=torch.double, device='cuda')

# 阻力参数 (64批次, 2维) - 包含线性和非线性阻力项
drag_2 = torch.randn((64, 2), dtype=torch.double, device='cuda')

# 俯仰控制延迟 (64批次, 1维) - 控制系统的时间延迟参数
pitch_ctl_delay = torch.randn((64, 1), dtype=torch.double, device='cuda')

# 标准重力加速度向量 - 地球重力在Z轴方向的标准值
g_std = torch.tensor([[0, 0, -9.80665]], dtype=torch.double, device='cuda')

# 预测执行器输出 (64批次, 3维) - 控制算法预测的推力向量
act_pred = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# 当前执行器输出 (64批次, 3维) - 当前时刻的实际推力向量
act = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# 位置向量 (64批次, 3维) - 无人机在3D空间中的位置
p = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# 速度向量 (64批次, 3维) - 无人机的3D速度
v = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# 风速向量 (64批次, 3维) - 环境风速对无人机的影响
v_wind = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# 加速度向量 (64批次, 3维) - 无人机的3D加速度
a = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

# ==================== 仿真参数设置 ====================
grad_decay = 0.4    # 梯度衰减系数，用于数值稳定性
ctl_dt = 1/15       # 控制时间步长 (秒)，对应15Hz的控制频率

def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    """
    PyTorch纯Python实现的四旋翼无人机前向仿真函数
    
    该函数实现了四旋翼无人机的物理仿真，包括控制器动态、空气阻力计算、
    以及运动学更新。用作CUDA实现的参考标准。
    
    参数:
        R (torch.Tensor): 旋转矩阵 [批次, 3, 3] - 描述无人机姿态
        dg (torch.Tensor): 重力偏差 [批次, 3] - 重力加速度变化
        z_drag_coef (torch.Tensor): Z轴阻力系数 [批次, 1] - 垂直阻力参数
        drag_2 (torch.Tensor): 阻力参数 [批次, 2] - 线性和非线性阻力项
        pitch_ctl_delay (torch.Tensor): 俯仰控制延迟 [批次, 1] - 控制时延
        act_pred (torch.Tensor): 预测执行器输出 [批次, 3] - 预测推力
        act (torch.Tensor): 当前执行器输出 [批次, 3] - 当前推力
        p (torch.Tensor): 位置 [批次, 3] - 3D位置坐标
        v (torch.Tensor): 速度 [批次, 3] - 3D速度向量
        v_wind (torch.Tensor): 风速 [批次, 3] - 环境风速
        a (torch.Tensor): 加速度 [批次, 3] - 3D加速度向量
        ctl_dt (float): 控制时间步长 - 仿真时间间隔
        
    返回:
        tuple: (act_next, p_next, v_next, a_next)
            - act_next: 下一时刻的执行器输出 [批次, 3]
            - p_next: 下一时刻的位置 [批次, 3]
            - v_next: 下一时刻的速度 [批次, 3]
            - a_next: 下一时刻的加速度 [批次, 3]
    """
    # 计算控制器延迟的指数衰减因子
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    
    # 更新执行器输出：使用一阶低通滤波器模拟控制延迟
    act_next = act_pred * (1 - alpha) + act * alpha
    
    # 计算相对风速在机体坐标系中的分量
    # v.add(-v_wind) 计算相对风速，然后转换到机体坐标系
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)
    
    # 计算空气阻力
    # 第一项：非线性阻力 (速度的平方项)
    # 0.047 = (4*rotor_drag_coefficient*motor_velocity_real) / sqrt(9.8)
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + 
                           v_left_s.abs() * v_left_s * R[..., 1] + 
                           v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    
    # 第二项：线性阻力
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + 
                            v_left_s * R[..., 1] + 
                            v_up_s * R[..., 2] * z_drag_coef)
    
    # 计算下一时刻的总加速度：推力 + 重力偏差 - 阻力
    a_next = act_next + dg - drag
    
    # 使用运动学方程更新位置：p = p0 + v*dt + 0.5*a*dt^2
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2
    
    # 使用运动学方程更新速度：v = v0 + (a + a_next)/2 * dt (梯形积分)
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt
    
    return act_next, p_next, v_next, a_next

# ==================== 前向传播测试 ====================
# 运行CUDA实现的前向传播
act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

# 运行PyTorch参考实现的前向传播
_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

# 验证CUDA实现与PyTorch实现的前向传播结果是否一致
assert torch.allclose(act_next, _act_next), "执行器输出不一致"
assert torch.allclose(a_next, _a_next), "加速度不一致"
assert torch.allclose(p_next, _p_next), "位置不一致"
assert torch.allclose(v_next, _v_next), "速度不一致"

print("前向传播测试通过：CUDA实现与PyTorch实现结果一致")

# ==================== 反向传播测试 ====================
# 生成随机的输出梯度，模拟损失函数对各输出的梯度
d_act_next = torch.randn_like(act_next)  # 对执行器输出的梯度
d_p_next = torch.randn_like(p_next)      # 对位置的梯度
d_v_next = torch.randn_like(v_next)      # 对速度的梯度
d_a_next = torch.randn_like(a_next)      # 对加速度的梯度

# 使用PyTorch自动求导计算参考梯度
torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),           # 输出张量
    (d_act_next, d_p_next, d_v_next, d_a_next),       # 对应的梯度
)

# 使用CUDA实现计算反向传播梯度
d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, 
    d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

# 验证CUDA实现的反向传播梯度与PyTorch自动求导结果是否一致
assert torch.allclose(d_act_pred, act_pred.grad), "act_pred梯度不一致"
assert torch.allclose(d_act, act.grad), "act梯度不一致"
assert torch.allclose(d_p, p.grad), "position梯度不一致"
assert torch.allclose(d_v, v.grad), "velocity梯度不一致"
assert torch.allclose(d_a, a.grad), "acceleration梯度不一致"

print("反向传播测试通过：CUDA实现与PyTorch自动求导结果一致")
print("所有测试通过！CUDA扩展模块工作正常。")
