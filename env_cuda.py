"""
DiffPhys-UAV环境模拟器 - CUDA版本

这是一个基于CUDA加速的无人机物理仿真环境，用于多智能体强化学习。
主要功能包括：
- 可微分的四旋翼物理模拟
- 多无人机群体协调
- 3D环境渲染
- 障碍物碰撞检测
- 梯度衰减机制

依赖:
    - torch: PyTorch深度学习框架
    - quadsim_cuda: 自定义CUDA内核模块
"""

import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function):
    """
    梯度衰减自定义函数
    
    这是一个PyTorch自动微分函数，用于在反向传播过程中对梯度进行衰减处理。
    在前向传播时直接返回输入，在反向传播时将梯度乘以衰减因子。
    
    主要用途：
    - 控制训练过程中的梯度大小
    - 防止梯度爆炸
    - 实现梯度的时间衰减
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向传播函数
        
        Args:
            ctx: PyTorch自动微分上下文，用于保存反向传播所需信息
            x (torch.Tensor): 输入张量，任意形状
            alpha (float): 梯度衰减因子，范围通常在(0, 1]之间
            
        Returns:
            torch.Tensor: 与输入x相同的张量（前向传播时不改变值）
        """
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播函数
        
        Args:
            ctx: 自动微分上下文，包含前向传播时保存的信息
            grad_output (torch.Tensor): 从上游传递的梯度
            
        Returns:
            tuple: (对x的梯度, 对alpha的梯度)
                - 对x的梯度: grad_output * alpha（应用衰减因子）
                - 对alpha的梯度: None（alpha不需要梯度）
        """
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    """
    四旋翼物理仿真自定义函数
    
    这是实现四旋翼物理仿真的核心自动微分函数，支持前向传播和反向传播。
    通过调用CUDA内核实现高性能的物理计算，包括动力学积分、空气阻力、
    控制延迟等复杂物理效应。
    
    主要特性：
    - CUDA加速的物理仿真
    - 可微分计算支持强化学习
    - 真实的四旋翼动力学模型
    - 环境扰动和噪声模拟
    """
    
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        """
        四旋翼仿真前向传播
        
        Args:
            ctx: PyTorch自动微分上下文
            R (torch.Tensor): 旋转矩阵 [batch_size, 3, 3] - 无人机姿态
            dg (torch.Tensor): 重力扰动 [batch_size, 3] - 重力噪声
            z_drag_coef (torch.Tensor): 垂直阻力系数 [batch_size, 1]
            drag_2 (torch.Tensor): 二次阻力系数 [batch_size, 2]
            pitch_ctl_delay (torch.Tensor): 俯仰控制延迟 [batch_size, 1]
            act_pred (torch.Tensor): 预测动作 [batch_size, 3] - 目标加速度
            act (torch.Tensor): 当前动作 [batch_size, 3] - 当前加速度
            p (torch.Tensor): 位置 [batch_size, 3] - xyz坐标
            v (torch.Tensor): 速度 [batch_size, 3] - xyz速度
            v_wind (torch.Tensor): 风速 [batch_size, 3] - 环境风场
            a (torch.Tensor): 加速度 [batch_size, 3] - 当前加速度
            grad_decay (float): 梯度衰减因子
            ctl_dt (float): 控制时间步长（秒）
            airmode (float): 空气模式参数
            
        Returns:
            tuple: (act_next, p_next, v_next, a_next)
                - act_next: 下一步动作 [batch_size, 3]
                - p_next: 下一步位置 [batch_size, 3] 
                - v_next: 下一步速度 [batch_size, 3]
                - a_next: 下一步加速度 [batch_size, 3]
        """
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        """
        四旋翼仿真反向传播
        
        计算输入参数相对于损失函数的梯度，支持端到端的强化学习训练。
        
        Args:
            ctx: 自动微分上下文，包含前向传播保存的张量
            d_act_next (torch.Tensor): 对下一步动作的梯度 [batch_size, 3]
            d_p_next (torch.Tensor): 对下一步位置的梯度 [batch_size, 3]
            d_v_next (torch.Tensor): 对下一步速度的梯度 [batch_size, 3]
            d_a_next (torch.Tensor): 对下一步加速度的梯度 [batch_size, 3]
            
        Returns:
            tuple: 对输入参数的梯度
                - None: R的梯度（不需要）
                - None: dg的梯度（不需要）
                - None: z_drag_coef的梯度（不需要）
                - None: drag_2的梯度（不需要）
                - None: pitch_ctl_delay的梯度（不需要）
                - d_act_pred: 对预测动作的梯度 [batch_size, 3]
                - d_act: 对当前动作的梯度 [batch_size, 3]
                - d_p: 对位置的梯度 [batch_size, 3]
                - d_v: 对速度的梯度 [batch_size, 3]
                - None: v_wind的梯度（不需要）
                - d_a: 对加速度的梯度 [batch_size, 3]
                - None: grad_decay的梯度（不需要）
                - None: ctl_dt的梯度（不需要）
                - None: airmode的梯度（不需要）
        """
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class Env:
    """
    多无人机仿真环境类
    
    这是一个支持多无人机协同的3D仿真环境，包含复杂的物理动力学、
    障碍物碰撞检测、环境渲染等功能。适用于强化学习和路径规划研究。
    
    主要特性：
    - 支持多无人机群体仿真
    - 可配置的3D障碍物环境
    - 实时渲染和可视化
    - 可微分物理仿真
    - 灵活的环境配置选项
    """
    
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        """
        初始化仿真环境
        
        Args:
            batch_size (int): 批次大小，同时仿真的环境数量
            width (int): 渲染图像宽度（像素）
            height (int): 渲染图像高度（像素）
            grad_decay (float): 梯度衰减因子，用于稳定训练 (0, 1]
            device (str, optional): 计算设备 ('cpu' 或 'cuda'). 默认: 'cpu'
            fov_x_half_tan (float, optional): 相机水平视场角的一半的正切值. 默认: 0.53
            single (bool, optional): 是否单无人机模式. 默认: False
            gate (bool, optional): 是否添加门形障碍物. 默认: False  
            ground_voxels (bool, optional): 是否添加地面体素障碍物. 默认: False
            scaffold (bool, optional): 是否添加脚手架结构. 默认: False
            speed_mtp (float, optional): 速度倍数，控制无人机最大速度. 默认: 1
            random_rotation (bool, optional): 是否随机旋转环境. 默认: False
            cam_angle (float, optional): 相机俯仰角（度）. 默认: 10
            
        Note:
            所有物理参数和障碍物配置都在此方法中初始化。环境坐标系为：
            - X轴：向右
            - Y轴：向前  
            - Z轴：向上
        """
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        
        # 球形障碍物参数 [x_range, y_range, z_range, radius_range]
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        
        # 长方体障碍物参数 [x_range, y_range, z_range, width, height, depth]
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        
        # 地面体素参数
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        
        # 圆柱形障碍物参数 [x_range, y_range, radius_range]
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        
        # 水平圆柱障碍物参数
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        
        # 门形障碍物参数 [x_range, y_range, z_range, radius_range]
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        
        # 风场参数
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        
        # 标准重力加速度
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        
        # 屋顶偏移参数
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        
        # 时间细分参数，用于碰撞检测
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        
        # 无人机初始位置配置（8个预设位置）
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 无人机目标位置配置
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 光流缓存
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        
        # 环境配置选项
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        
        # 初始化环境状态
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        """
        重置环境状态
        
        重新初始化所有环境参数，包括障碍物位置、无人机状态、相机配置等。
        每个episode开始时调用此方法生成新的随机环境配置。
        
        主要功能：
        - 随机生成障碍物分布（球体、长方体、圆柱等）
        - 初始化无人机物理状态（位置、速度、姿态）
        - 配置相机和渲染参数
        - 设置环境变量（风场、重力扰动等）
        - 根据配置选项添加特殊结构（门、脚手架等）
        
        Returns:
            None: 直接修改类实例状态
        """
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # scaffold
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        """
        更新无人机姿态向量
        
        根据推力向量和预测速度更新无人机的三轴姿态矩阵。
        实现了四旋翼的姿态控制逻辑，包括偏航惯性和姿态平滑过渡。
        
        Args:
            R (torch.Tensor): 当前旋转矩阵 [batch_size, 3, 3]
            a_thr (torch.Tensor): 推力加速度向量 [batch_size, 3] (包含重力)
            v_pred (torch.Tensor): 预测速度向量 [batch_size, 3]
            alpha (float): 姿态平滑因子 [0, 1]，控制姿态变化速度
            yaw_inertia (float, optional): 偏航惯性系数. 默认: 5
            
        Returns:
            torch.Tensor: 更新后的旋转矩阵 [batch_size, 3, 3]
                矩阵列向量分别为：[前向量, 左向量, 上向量]
                
        Note:
            - 前向量：无人机机头方向
            - 左向量：无人机左侧方向  
            - 上向量：由推力方向决定
            - 所有向量都是单位向量且相互正交
        """
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        
        # 去除重力影响，获得纯推力向量
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        
        # 上向量由推力方向决定
        self_up_vec = a_thr / thrust
        
        # 计算新的前向量，考虑偏航惯性
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        
        # 确保前向量与上向量正交
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        
        # 左向量通过叉积计算，确保正交性
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        """
        渲染3D环境场景
        
        从无人机第一人称视角渲染环境，生成深度图像用于视觉感知。
        使用CUDA加速的光线追踪算法实现实时渲染。
        
        Args:
            ctl_dt (float): 控制时间步长，用于运动模糊等效果
            
        Returns:
            tuple: (canvas, flow)
                - canvas (torch.Tensor): 渲染的深度图像 [batch_size, height, width]
                - flow: 光流信息（当前返回None）
                
        Note:
            渲染考虑了以下元素：
            - 球形障碍物 (balls)
            - 圆柱形障碍物 (cyl, cyl_h)  
            - 长方体障碍物 (voxels)
            - 无人机当前和历史位置（用于运动模糊）
            - 相机视场角和姿态
        """
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        """
        计算到最近障碍物的向量
        
        在无人机预测轨迹上的多个时间点计算到最近障碍物表面的向量，
        用于碰撞检测和避障规划。
        
        Returns:
            torch.Tensor: 到最近点的向量 [sub_div_steps, batch_size, 3]
                向量指向最近的障碍物表面点，长度为距离
                
        Note:
            - 使用sub_div进行时间细分，提高碰撞检测精度
            - 考虑无人机半径，确保安全距离
            - 检测所有类型的障碍物（球体、圆柱、长方体）
        """
        # 在预测轨迹上的多个时间点计算位置
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        
        # 调用CUDA内核计算最近点
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, 
                                   self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        执行一步物理仿真
        
        根据预测动作更新无人机状态，包括位置、速度、加速度和姿态。
        使用可微分物理引擎支持端到端训练。
        
        Args:
            act_pred (torch.Tensor): 预测动作/目标加速度 [batch_size, 3]
            ctl_dt (float, optional): 控制时间步长（秒）. 默认: 1/15
            v_pred (torch.Tensor, optional): 预测速度，用于姿态更新 [batch_size, 3]
            
        Returns:
            None: 直接更新类实例的状态变量
            
        Note:
            更新的状态包括：
            - self.act: 当前动作/加速度
            - self.p: 位置
            - self.v: 速度  
            - self.a: 加速度
            - self.R: 姿态矩阵
            - self.dg: 重力扰动（随机游走更新）
        """
        # 更新重力扰动（随机游走过程）
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        
        # 保存当前位置作为历史位置
        self.p_old = self.p
        
        # 调用可微分物理仿真
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
            
        # 更新姿态矩阵
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        执行一步物理仿真（Python实现版本）
        
        这是run方法的Python实现版本，不使用CUDA内核。
        主要用于调试和对比验证，包含完整的四旋翼动力学模型。
        
        Args:
            act_pred (torch.Tensor): 预测动作/目标加速度 [batch_size, 3]
            ctl_dt (float, optional): 控制时间步长（秒）. 默认: 1/15
            v_pred (torch.Tensor, optional): 预测速度，用于姿态更新 [batch_size, 3]
            
        Returns:
            None: 直接更新类实例的状态变量
            
        Note:
            物理模型包括：
            - 一阶控制延迟
            - 重力扰动（随机游走）
            - 垂直阻力（与电机转速相关）
            - 二次空气阻力
            - 梯度衰减机制
            - 运动学积分（位置、速度、加速度）
        """
        # 应用控制延迟
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        
        # 更新重力扰动
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        
        # 计算垂直阻力（与电机转速相关）
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
            
        # 计算二次空气阻力
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        
        # 计算下一步加速度
        a_next = self.act + self.dg - z_drag - drag
        
        # 更新运动学状态
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # 更新姿态
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

