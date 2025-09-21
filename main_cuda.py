"""
DiffPhys-UAV主训练脚本 - CUDA版本

这是一个基于可微分物理仿真的多无人机强化学习训练脚本。
使用端到端的可微分物理引擎训练神经网络控制器，实现无人机的自主导航和避障。

主要功能：
- 多无人机协同训练
- 可微分物理仿真
- 视觉感知（深度图像）
- 碰撞避障
- 速度和轨迹控制
- 损失函数设计和优化

训练目标：
- 学习从目标位置导航到目的地
- 避免与障碍物碰撞
- 保持平滑的飞行轨迹
- 速度控制和能耗优化

依赖：
- PyTorch: 深度学习框架
- env_cuda: 可微分无人机仿真环境
- model: 神经网络控制器模型
- tensorboard: 训练监控和可视化
"""

from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


# ======================== 命令行参数配置 ========================
"""
训练超参数和环境配置
支持通过命令行参数调整所有关键训练设置
"""
parser = argparse.ArgumentParser(description='DiffPhys-UAV 多无人机强化学习训练')

# 基础训练参数
parser.add_argument('--resume', default=None, help='恢复训练的模型检查点路径')
parser.add_argument('--batch_size', type=int, default=64, help='批次大小，同时训练的环境数量')
parser.add_argument('--num_iters', type=int, default=50000, help='总训练迭代次数')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减因子')

# 损失函数权重系数
parser.add_argument('--coef_v', type=float, default=1.0, 
                   help='速度跟踪损失权重：smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, 
                   help='速度损失权重（已废弃）')
parser.add_argument('--coef_v_pred', type=float, default=2.0, 
                   help='速度预测损失权重：用于无里程计模式的速度估计')
parser.add_argument('--coef_collide', type=float, default=2.0, 
                   help='碰撞损失权重：接近障碍物时损失增大，否则为零')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, 
                   help='避障损失权重：二次距离间隙损失')
parser.add_argument('--coef_d_acc', type=float, default=0.01, 
                   help='加速度正则化权重：控制动作平滑性')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, 
                   help='急动度正则化权重：控制加速度变化率')
parser.add_argument('--coef_d_snap', type=float, default=0.0, 
                   help='急动度二阶导数权重（已废弃）')
parser.add_argument('--coef_ground_affinity', type=float, default=0., 
                   help='地面亲和力权重（已废弃）')
parser.add_argument('--coef_bias', type=float, default=0.0, 
                   help='偏置损失权重（已废弃）')

# 环境和仿真参数
parser.add_argument('--speed_mtp', type=float, default=1.0, 
                   help='速度倍数：控制无人机最大飞行速度')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53, 
                   help='相机水平视场角的一半的正切值')
parser.add_argument('--timesteps', type=int, default=150, 
                   help='每个episode的时间步数')
parser.add_argument('--cam_angle', type=int, default=10, 
                   help='相机俯仰角（度）')

# 环境特性开关
parser.add_argument('--single', default=False, action='store_true', 
                   help='单无人机模式（默认多无人机）')
parser.add_argument('--gate', default=False, action='store_true', 
                   help='添加门形障碍物')
parser.add_argument('--ground_voxels', default=False, action='store_true', 
                   help='添加地面体素障碍物')
parser.add_argument('--scaffold', default=False, action='store_true', 
                   help='添加脚手架结构')
parser.add_argument('--random_rotation', default=False, action='store_true', 
                   help='随机旋转环境布局')
parser.add_argument('--yaw_drift', default=False, action='store_true', 
                   help='启用偏航漂移扰动')
parser.add_argument('--no_odom', default=False, action='store_true', 
                   help='无里程计模式：不使用速度反馈')

# 解析命令行参数并初始化
args = parser.parse_args()
writer = SummaryWriter()  # TensorBoard日志记录器
print(args)

# ======================== 设备和环境初始化 ========================
device = torch.device('cuda')  # 使用CUDA加速

# 创建仿真环境
# 64x48像素深度图像输入，支持多种环境配置
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)

# 创建神经网络模型
# 输入：视觉特征(7维) + 状态特征(3维里程计或0维无里程计)
# 输出：控制动作(6维：3维加速度 + 3维速度预测)
if args.no_odom:
    model = Model(7, 6)  # 无里程计模式：7维状态输入
else:
    model = Model(7+3, 6)  # 有里程计模式：10维状态输入（包含3维速度）
model = model.to(device)

# ======================== 模型加载和优化器配置 ========================
# 恢复训练（如果指定了检查点）
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)

# 优化器配置：使用AdamW优化器和余弦退火学习率调度
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

# 控制时间步长（15 FPS）
ctl_dt = 1 / 15

# ======================== 辅助函数定义 ========================

# 训练指标平滑记录器
scaler_q = defaultdict(list)

def smooth_dict(ori_dict):
    """
    训练指标平滑记录函数
    
    将训练过程中的各种指标值添加到缓冲区，用于后续的平滑显示和日志记录。
    
    Args:
        ori_dict (dict): 包含指标名称和数值的字典
                        例: {'loss': 0.5, 'success': 0.8}
                        
    Returns:
        None: 直接修改全局变量scaler_q
        
    Note:
        指标值会被转换为float类型并存储在对应的列表中
    """
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    """
    障碍物避让损失函数（屏障函数）
    
    计算基于距离的二次惩罚损失，当无人机接近障碍物时施加较大惩罚。
    使用屏障函数确保无人机保持安全距离。
    
    Args:
        x (torch.Tensor): 标准化距离 [时间步, 批次] 
                         x=1表示安全距离，x<1表示过于接近
        v_to_pt (torch.Tensor): 接近速度权重 [时间步, 批次]
                               较大的接近速度会增加惩罚
                               
    Returns:
        torch.Tensor: 标量损失值
                     当x<1时产生二次惩罚，x>=1时为0
                     
    Note:
        使用ReLU确保只对距离不足的情况进行惩罚
        二次惩罚函数形式：(v_to_pt * (1-x)^2) 当 x<1
    """
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    """
    判断是否为保存迭代
    
    根据训练迭代次数决定是否保存模型和可视化结果。
    训练初期保存频率较高，后期保存频率降低。
    
    Args:
        i (int): 当前迭代次数（从0开始）
        
    Returns:
        bool: True表示应该保存，False表示跳过
        
    Note:
        保存策略：
        - 前2000次迭代：每250次保存一次
        - 2000次之后：每1000次保存一次
    """
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

# ======================== 主训练循环 ========================
pbar = tqdm(range(args.num_iters), ncols=80)  # 进度条
B = args.batch_size

for i in pbar:
    # -------------------- Episode 初始化 --------------------
    """
    每个训练迭代开始时重置环境和模型状态，
    初始化数据收集列表和控制参数
    """
    env.reset()  # 重置环境：生成新的障碍物布局和无人机初始状态
    model.reset()  # 重置模型的隐藏状态（RNN/LSTM）
    
    # 数据收集列表：记录整个episode的轨迹和状态
    p_history = []          # 位置历史
    v_history = []          # 速度历史  
    target_v_history = []   # 目标速度历史
    vec_to_pt_history = []  # 到最近障碍物的向量历史
    act_diff_history = []   # 动作差分历史（未使用）
    v_preds = []           # 速度预测历史
    vid = []               # 视频帧（用于可视化）
    v_net_feats = []       # 网络特征历史（未使用）
    h = None               # RNN隐藏状态
    
    # 控制延迟缓冲区：模拟真实无人机的执行器延迟
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    
    # 计算初始目标速度向量
    target_v_raw = env.p_target - env.p
    
    # 偏航漂移配置（可选）：模拟无人机偏航系统的累积误差
    if args.yaw_drift:
        # 每步5度/秒的随机偏航漂移
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    # -------------------- 时间步仿真循环 --------------------
    """
    在每个episode中执行固定时间步数的仿真，
    收集感知数据、执行控制决策、更新物理状态
    """
    for t in range(args.timesteps):
        # 随机化控制时间步长：模拟真实系统的时序不确定性
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        
        # 环境感知：从无人机视角渲染深度图像
        depth, flow = env.render(ctl_dt)
        
        # 记录当前状态用于损失计算
        p_history.append(env.p.clone())
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())
        
        # 保存可视化帧（仅在保存迭代时）
        if is_save_iter(i):
            vid.append(depth[4])  # 保存第4个样本的深度图
        
        # 更新目标速度向量
        if args.yaw_drift:
            # 应用偏航漂移：模拟导航系统累积误差
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            # 标准模式：直接计算到目标的向量
            target_v_raw = env.p_target - env.p.detach()
            
        # 执行物理仿真步骤
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # -------------------- 状态空间变换 --------------------
        """
        将世界坐标系的状态转换为无人机本体坐标系，
        构建神经网络的输入特征向量
        """
        # 构建水平旋转矩阵：忽略俯仰和滚转，只考虑偏航
        R = env.R
        fwd = env.R[:, :, 0].clone()  # 前向量
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0          # 投影到水平面
        up[:, 2] = 1           # 向上向量
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        # 计算限速后的目标速度
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        
        # 构建神经网络输入状态向量
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),  # 本体坐标系下的目标速度 [3D]
            env.R[:, 2],                              # 当前姿态的上向量 [3D]
            env.margin[:, None]]                      # 安全边距 [1D]
        
        # 当前速度（本体坐标系）
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        
        # 可选：添加里程计信息
        if not args.no_odom:
            state.insert(0, local_v)  # 在状态向量开头插入速度信息
            
        state = torch.cat(state, -1)  # 拼接所有状态特征

        # -------------------- 视觉特征处理 --------------------
        """
        对深度图像进行预处理，转换为神经网络可接受的格式
        """
        # 深度图像归一化和噪声注入
        # 将深度值转换为 [0.3, 24] 米范围，然后归一化并添加噪声
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)  # 下采样：64x48 -> 16x12
        
        # -------------------- 神经网络推理 --------------------
        """
        使用循环神经网络处理视觉和状态信息，
        输出控制动作和价值估计
        """
        act, values, h = model(x, state, h)
        
        # -------------------- 动作空间变换 --------------------
        """
        将网络输出的本体坐标系动作转换为世界坐标系，
        并应用推力估计误差补偿
        """
        # 从本体坐标系转换到世界坐标系
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred.clone())
        
        # 推力估计误差补偿：模拟真实无人机的推力不准确性
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        
        # 记录网络特征（用于分析，当前未使用）
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        # 记录历史数据用于损失计算
        v_history.append(env.v.clone())
        target_v_history.append(target_v.clone())

    # -------------------- Episode 结束后的损失计算 --------------------
    """
    使用收集的轨迹数据计算各种损失函数
    """
    p_history = torch.stack(p_history)
    # 地面亲和力损失：防止无人机飞到地面以下
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    # -------------------- 速度跟踪损失 --------------------
    """
    计算无人机实际速度与目标速度的差异
    使用滑动窗口平均来减少瞬时噪声影响
    """
    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    # 30步滑动窗口平均：平滑瞬时速度波动
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    
    # 计算平均速度与目标速度的差异
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    # -------------------- 速度预测损失 --------------------
    """
    网络自监督学习：预测下一步速度
    提高网络对动力学的理解能力
    """
    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    # -------------------- 方向偏置损失 --------------------
    """
    确保无人机主要沿目标方向飞行，减少不必要的侧向运动
    """
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    # 计算实际速度在目标方向上的投影
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    # 惩罚偏离目标方向的速度分量
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    # -------------------- 控制平滑性损失 --------------------
    """
    正则化控制动作，确保飞行轨迹平滑，减少能耗
    """
    # 急动度损失：加速度的时间导数（控制平滑性）
    jerk_history = act_buffer.diff(1, 0).mul(15)
    # 急动度二阶导数：加速度方向变化率
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    
    # 各种正则化损失
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()    # 总加速度大小
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean() # 急动度大小
    loss_d_snap = snap_history.pow(2).sum(-1).mean() # 急动度变化率

    # -------------------- 碰撞避障损失 --------------------
    """
    基于与障碍物的距离计算避障损失
    距离越近、接近速度越快，损失越大
    """
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin  # 减去安全边距
    
    # 计算接近速度（用于动态调整惩罚强度）
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    
    # 屏障函数：距离不足时施加二次惩罚
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    
    # 碰撞损失：使用softplus函数，距离越近损失指数增长
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    # -------------------- 速度大小损失 --------------------
    """
    确保无人机在目标方向上的速度与期望速度匹配
    """
    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    # -------------------- 总损失函数 --------------------
    """
    加权组合所有损失项，形成最终的训练目标
    """
    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity * loss_ground_affinity

    # 异常检测：防止训练崩溃
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    # -------------------- 梯度更新 --------------------
    """
    执行反向传播和参数更新
    """
    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()  # 更新学习率

    # -------------------- 训练指标统计和记录 --------------------
    """
    计算训练指标并记录到TensorBoard
    """
    with torch.no_grad():
        # 性能指标计算
        avg_speed = speed_history.mean(0)
        # 成功率：整个episode都没有碰撞的比例
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        
        # 记录所有训练指标
        smooth_dict({
            'loss': loss,                           # 总损失
            'loss_v': loss_v,                       # 速度跟踪损失
            'loss_v_pred': loss_v_pred,             # 速度预测损失
            'loss_obj_avoidance': loss_obj_avoidance, # 避障损失
            'loss_d_acc': loss_d_acc,               # 加速度正则化
            'loss_d_jerk': loss_d_jerk,             # 急动度正则化
            'loss_d_snap': loss_d_snap,             # 急动度变化率正则化
            'loss_bias': loss_bias,                 # 方向偏置损失
            'loss_speed': loss_speed,               # 速度大小损失
            'loss_collide': loss_collide,           # 碰撞损失
            'loss_ground_affinity': loss_ground_affinity, # 地面亲和力损失
            'success': _success,                    # 成功率（无碰撞完成任务）
            'max_speed': speed_history.max(0).values.mean(), # 最大速度
            'avg_speed': avg_speed.mean(),          # 平均速度
            'ar': (success * avg_speed).mean()})    # 成功加权平均速度

        # -------------------- 可视化和模型保存 --------------------
        """
        定期保存训练结果的可视化图表和模型检查点
        """
        if is_save_iter(i):
            # 创建位置轨迹图
            fig_p, ax = plt.subplots()
            p_history_sample = p_history[:, 4].cpu()  # 选择第4个样本
            ax.plot(p_history_sample[:, 0], label='x')
            ax.plot(p_history_sample[:, 1], label='y') 
            ax.plot(p_history_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Position Trajectory')
            
            # 创建速度轨迹图
            fig_v, ax = plt.subplots()
            v_history_sample = v_history[:, 4].cpu()
            ax.plot(v_history_sample[:, 0], label='x')
            ax.plot(v_history_sample[:, 1], label='y')
            ax.plot(v_history_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Velocity Trajectory')
            
            # 创建控制动作图
            fig_a, ax = plt.subplots()
            act_buffer_sample = act_buffer[:, 4].cpu()
            ax.plot(act_buffer_sample[:, 0], label='x')
            ax.plot(act_buffer_sample[:, 1], label='y')
            ax.plot(act_buffer_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Control Actions')
            
            # 将图表记录到TensorBoard
            # writer.add_video('demo', vid, i + 1, 15)  # 可选：保存视频
            writer.add_figure('p_history', fig_p, i + 1)  # 位置轨迹
            writer.add_figure('v_history', fig_v, i + 1)  # 速度轨迹
            writer.add_figure('a_reals', fig_a, i + 1)    # 控制动作
            
        # 定期保存模型检查点
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
            
        # 定期记录平滑后的训练指标
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()  # 清空缓冲区
