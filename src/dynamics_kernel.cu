
/**
 * @file dynamics_kernel.cu
 * @brief 无人机动力学仿真的CUDA加速核函数实现
 * 
 * 本文件包含了用于四旋翼无人机动力学仿真的CUDA核函数，主要用于：
 * 1. 无人机姿态状态向量的更新计算
 * 2. 动力学前向传播，包括位置、速度、加速度的时间积分
 * 3. 反向传播梯度计算，支持基于梯度的优化和机器学习
 * 
 * 主要功能模块：
 * - 执行器延迟建模：模拟真实无人机的执行器响应延迟
 * - 空气阻力计算：包括线性和二次阻力项
 * - 风场影响：考虑外部风速对无人机运动的影响
 * - 姿态控制：基于推力向量和速度预测更新旋转矩阵
 * - 自动微分：提供完整的反向传播支持
 * 
 * 该实现使用CUDA并行计算来加速批量无人机的动力学仿真，
 * 特别适用于强化学习和控制算法的训练场景。
 * 
 * @author DiffPhys-UAV项目组
 * @date 2025年9月21日
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

/**
 * @brief CUDA核函数：更新无人机的状态向量和旋转矩阵
 * 
 * 根据推力向量和预测速度更新无人机的状态向量，包括前进方向、左侧方向和上方向向量。
 * 主要用于计算无人机的姿态矩阵R，该矩阵定义了无人机在世界坐标系中的方向。
 * 
 * @param R_new      [输出] 更新后的旋转矩阵 (B x 3 x 3)，其中：
 *                          R_new[b][0][:] = 前进方向向量 (forward vector)
 *                          R_new[b][1][:] = 左侧方向向量 (left vector)  
 *                          R_new[b][2][:] = 上方向向量 (up vector)
 * @param R          [输入] 当前的旋转矩阵 (B x 3 x 3)
 * @param a_thr      [输入] 推力加速度向量 (B x 3)，不包括重力
 * @param v_pred     [输入] 预测的速度向量 (B x 3)
 * @param alpha      [输入] 偏航惯性系数 (B x 1)，控制前进方向的变化率
 * @param yaw_inertia [输入] 偏航惯性参数，用于平滑方向变化
 */
template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_thr,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> alpha,
    float yaw_inertia) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;
    
    // 步骤1: 计算推力向量，添加重力补偿
    // a_thr = a_thr - self.g_std; (在调用前已处理)
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665;  // 添加重力加速度
    
    // 步骤2: 计算推力大小和上方向向量（单位向量）
    // thrust = torch.norm(a_thr, 2, -1, True);
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    // self.up_vec = a_thr / thrust;
    scalar_t ux = ax / thrust;  // 上方向向量x分量
    scalar_t uy = ay / thrust;  // 上方向向量y分量  
    scalar_t uz = az / thrust;  // 上方向向量z分量
    
    // 步骤3: 计算前进方向向量，结合偏航惯性和预测速度
    // forward_vec = self.forward_vec * yaw_inertia + v_pred;
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    
    // 步骤4: 归一化前进方向向量，并与当前前进方向混合
    // forward_vec = F.normalize(forward_vec, 2, -1);
    // forward_vec = (1-alpha) * forward_vec + alpha * self.forward_vec
    scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];
    
    // 步骤5: 确保前进方向与上方向垂直（投影到水平面）
    // forward_vec[2] = (forward_vec[0] * self_up_vec[0] + forward_vec[1] * self_up_vec[1]) / -self_up_vec[2]
    fz = (fx * ux + fy * uy) / -uz;
    
    // 步骤6: 再次归一化前进方向向量
    // self.forward_vec = F.normalize(forward_vec, 2, -1);
    t = sqrt(fx * fx + fy * fy + fz * fz);
    fx /= t;
    fy /= t;
    fz /= t;
    
    // 步骤7: 构建旋转矩阵，使用叉乘计算左侧方向向量
    // self.left_vec = torch.cross(self.up_vec, self.forward_vec);
    R_new[b][0][0] = fx;                    // 前进方向向量
    R_new[b][0][1] = uy * fz - uz * fy;     // 左侧方向向量 = up × forward
    R_new[b][0][2] = ux;                    // 上方向向量
    R_new[b][1][0] = fy;
    R_new[b][1][1] = uz * fx - ux * fz;
    R_new[b][1][2] = uy;
    R_new[b][2][0] = fz;
    R_new[b][2][1] = ux * fy - uy * fx;
    R_new[b][2][2] = uz;
}


/**
 * @brief CUDA核函数：无人机动力学前向传播计算
 * 
 * 根据当前状态和控制输入计算下一时刻的无人机状态，包括位置、速度、加速度和执行器状态。
 * 该函数实现了完整的四旋翼动力学模型，包括空气阻力、风场影响、执行器延迟等效应。
 * 
 * @param R               [输入] 当前旋转矩阵 (B x 3 x 3)，定义无人机姿态
 * @param dg              [输入] 重力扰动/噪声 (B x 3)
 * @param z_drag_coef     [输入] 垂直方向阻力系数 (B x 1)
 * @param drag_2          [输入] 阻力系数 [二次项系数, 一次项系数] (B x 2)
 * @param pitch_ctl_delay [输入] 俯仰控制延迟参数 (B x 1)
 * @param act_pred        [输入] 预测的执行器输出/控制指令 (B x 3)
 * @param act             [输入] 当前执行器状态 (B x 3)
 * @param p               [输入] 当前位置 (B x 3)
 * @param v               [输入] 当前速度 (B x 3)
 * @param v_wind          [输入] 风速向量 (B x 3)
 * @param a               [输入] 当前加速度 (B x 3)
 * @param act_next        [输出] 下一时刻执行器状态 (B x 3)
 * @param p_next          [输出] 下一时刻位置 (B x 3)
 * @param v_next          [输出] 下一时刻速度 (B x 3)
 * @param a_next          [输出] 下一时刻加速度 (B x 3)
 * @param ctl_dt          [输入] 控制时间步长
 * @param airmode_av2a    [输入] 空中模式角速度到加速度的转换系数
 */
template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,
    float ctl_dt, float airmode_av2a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // 步骤1: 计算执行器延迟效应
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    // self.act = act_pred * (1 - alpha) + self.act * alpha
    // 使用一阶低通滤波器模拟执行器响应延迟
    for (int j=0; j<3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    
    // 步骤2: 计算相对风速（无人机速度相对于风速）
    // self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
    // v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    
    // 步骤3: 将相对风速投影到无人机坐标系
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];     // 上方向分量
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];    // 前进方向分量
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];   // 左侧方向分量
    
    // 步骤4: 计算阻力项（二次和一次项）
    // 使用速度的平方项和一次项来模拟空气阻力
    scalar_t v_up_2 = v_up_s * abs(v_up_s);      // 上方向二次阻力项
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);   // 前进方向二次阻力项
    scalar_t v_left_2 = v_left_s * abs(v_left_s); // 左侧方向二次阻力项

    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j=0; j<3; j++){
        // 二次阻力项：在各个方向上的阻力加速度
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        // 一次阻力项：线性阻力
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }
    
    // 步骤5: 计算空中模式效应（airmode）
    // 根据执行器状态变化计算角速度，并转换为加速度补偿
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;  // 角速度

    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,  // 空中模式补偿加速度
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a};
    
    // 步骤6: 计算下一时刻的加速度
    // motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
    // z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
    // a_next = self.act + self.dg - z_drag
    for (int j=0; j<3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
    
    // 步骤7: 使用中点法更新位置和速度
    // self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
    for (int j=0; j<3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    // self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
    for (int j=0; j<3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}


/**
 * @brief CUDA核函数：无人机动力学反向传播计算
 * 
 * 计算前向传播过程的梯度，用于基于梯度的优化和机器学习。
 * 该函数实现了run_forward_cuda_kernel的反向传播，计算损失函数对输入变量的梯度。
 * 
 * @param R               [输入] 旋转矩阵 (B x 3 x 3)
 * @param dg              [输入] 重力扰动 (B x 3)  
 * @param z_drag_coef     [输入] 垂直阻力系数 (B x 1)
 * @param drag_2          [输入] 阻力系数 (B x 2)
 * @param pitch_ctl_delay [输入] 俯仰控制延迟 (B x 1)
 * @param v               [输入] 当前速度 (B x 3)
 * @param v_wind          [输入] 风速 (B x 3)
 * @param act_next        [输入] 下一时刻执行器状态 (B x 3)
 * @param d_act_pred      [输出] 预测执行器输出的梯度 (B x 3)
 * @param d_act           [输出] 当前执行器状态的梯度 (B x 3)
 * @param d_p             [输出] 当前位置的梯度 (B x 3)
 * @param d_v             [输出] 当前速度的梯度 (B x 3)
 * @param d_a             [输出] 当前加速度的梯度 (B x 3)
 * @param _d_act_next     [输入] 下一时刻执行器状态的梯度 (B x 3)
 * @param d_p_next        [输入] 下一时刻位置的梯度 (B x 3)
 * @param d_v_next        [输入] 下一时刻速度的梯度 (B x 3)
 * @param _d_a_next       [输入] 下一时刻加速度的梯度 (B x 3)
 * @param grad_decay      [输入] 梯度衰减系数
 * @param ctl_dt          [输入] 控制时间步长
 */
template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,
    float grad_decay,
    float ctl_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // 重复前向传播中的一些计算，用于反向传播
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // 从输入梯度中复制局部变量
    scalar_t d_act_next[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    
    // 反向传播开始：按照前向传播的逆序计算梯度
    // 步骤1: 速度更新的反向传播
    // v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
    for (int j=0; j<3; j++){
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);  // 应用梯度衰减
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];            // 当前加速度的梯度
        d_a_next[j] += 0.5 * ctl_dt * d_v_next[i][j];         // 下一时刻加速度的梯度
    }
    
    // 步骤2: 位置更新的反向传播
    // p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    for (int j=0; j<3; j++){
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt); // 当前位置的梯度
        d_v[i][j] += ctl_dt * d_p_next[i][j];                 // 速度对位置的贡献
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j]; // 加速度对位置的贡献
    }
    
    // 步骤3: 加速度计算的反向传播
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    for (int j=0; j<3; j++){
        // a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
        d_act_next[j] += d_a_next[j];                         // 执行器状态的梯度
        d_a_drag_2[j] = -d_a_next[j] * drag_2[i][0];         // 二次阻力项的梯度
        d_a_drag_1[j] = -d_a_next[j] * drag_2[i][1];         // 一次阻力项的梯度
    }

    // 步骤4: 阻力计算的反向传播
    // 重新计算相对风速和投影
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    
    // 阻力项的反向传播
    for (int j=0; j<3; j++){
        // a_drag_2[j] = v_up_s * v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * v_left_s * R[i][j][1] + v_fwd_s * v_fwd_s * R[i][j][0];
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];   // 前进方向二次项梯度
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1]; // 左侧方向二次项梯度
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0]; // 上方向二次项梯度
        
        // a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];               // 前进方向一次项梯度
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];              // 左侧方向一次项梯度
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0]; // 上方向一次项梯度
    }

    // 步骤5: 速度投影的反向传播
    // 将各方向速度分量的梯度转换回世界坐标系
    for (int j=0; j<3; j++){
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;   // 前进方向的贡献
        d_v[i][j] += R[i][j][1] * d_v_left_s;  // 左侧方向的贡献
        d_v[i][j] += R[i][j][2] * d_v_up_s;    // 上方向的贡献
    }
    
    // 步骤6: 执行器延迟的反向传播
    for (int j=0; j<3; j++){
        // act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
        d_act_pred[i][j] = (1 - alpha) * d_act_next[j];  // 预测执行器输出的梯度
        d_act[i][j] = alpha * d_act_next[j];             // 当前执行器状态的梯度
    }
}

} // namespace

/**
 * @brief C++接口函数：无人机动力学前向传播
 * 
 * 这是一个C++到CUDA的接口函数，用于从Python/PyTorch调用CUDA核函数进行无人机动力学计算。
 * 该函数封装了CUDA核函数的调用，处理张量的内存管理和GPU线程配置。
 * 
 * @param R               当前旋转矩阵 (B x 3 x 3)
 * @param dg              重力扰动向量 (B x 3)
 * @param z_drag_coef     垂直阻力系数 (B x 1)
 * @param drag_2          阻力系数 [二次项, 一次项] (B x 2)
 * @param pitch_ctl_delay 俯仰控制延迟参数 (B x 1)
 * @param act_pred        预测的执行器输出 (B x 3)
 * @param act             当前执行器状态 (B x 3)
 * @param p               当前位置 (B x 3)
 * @param v               当前速度 (B x 3)
 * @param v_wind          风速向量 (B x 3)
 * @param a               当前加速度 (B x 3)
 * @param ctl_dt          控制时间步长
 * @param airmode_av2a    空中模式角速度到加速度转换系数
 * 
 * @return std::vector<torch::Tensor> 包含 [act_next, p_next, v_next, a_next] 的向量
 */
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a){

    // 创建输出张量，与输入张量具有相同的形状和设备
    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    // 配置CUDA执行参数：每个批次元素对应一个线程
    const int threads = R.size(0);  // 批次大小
    const dim3 blocks(1);           // 单个块
    
    // 调用CUDA核函数，使用模板分发处理不同的数据类型
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt, airmode_av2a);
    }));
    
    // 返回计算结果
    return {act_next, p_next, v_next, a_next};
}


/**
 * @brief C++接口函数：无人机动力学反向传播
 * 
 * 这是一个C++到CUDA的接口函数，用于计算前向传播的梯度。
 * 该函数实现了自动微分中的反向传播，为基于梯度的优化提供必要的梯度信息。
 * 
 * @param R               旋转矩阵 (B x 3 x 3)
 * @param dg              重力扰动 (B x 3)
 * @param z_drag_coef     垂直阻力系数 (B x 1)
 * @param drag_2          阻力系数 (B x 2)
 * @param pitch_ctl_delay 俯仰控制延迟 (B x 1)
 * @param v               速度 (B x 3)
 * @param v_wind          风速 (B x 3)
 * @param act_next        下一时刻执行器状态 (B x 3)
 * @param _d_act_next     下一时刻执行器状态的梯度 (B x 3)
 * @param d_p_next        下一时刻位置的梯度 (B x 3)
 * @param d_v_next        下一时刻速度的梯度 (B x 3)
 * @param _d_a_next       下一时刻加速度的梯度 (B x 3)
 * @param grad_decay      梯度衰减系数
 * @param ctl_dt          控制时间步长
 * 
 * @return std::vector<torch::Tensor> 包含 [d_act_pred, d_act, d_p, d_v, d_a] 的向量
 */
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt){

    // 创建输出梯度张量
    torch::Tensor d_act_pred = torch::empty_like(dg);  // 预测执行器输出的梯度
    torch::Tensor d_act = torch::empty_like(dg);       // 当前执行器状态的梯度
    torch::Tensor d_p = torch::empty_like(dg);         // 当前位置的梯度
    torch::Tensor d_v = torch::empty_like(dg);         // 当前速度的梯度
    torch::Tensor d_a = torch::empty_like(dg);         // 当前加速度的梯度

    // 配置CUDA执行参数
    const int threads = R.size(0);  // 批次大小
    const dim3 blocks(1);           // 单个块
    
    // 调用CUDA核函数进行反向传播计算
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, ctl_dt);
    }));
    
    // 返回计算的梯度
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

/**
 * @brief C++接口函数：更新无人机状态向量
 * 
 * 这是一个C++到CUDA的接口函数，用于更新无人机的姿态旋转矩阵。
 * 根据推力向量和预测速度计算新的旋转矩阵，该矩阵定义了无人机的空间方向。
 * 
 * @param R          当前旋转矩阵 (B x 3 x 3)
 * @param a_thr      推力加速度向量 (B x 3)，不包括重力
 * @param v_pred     预测速度向量 (B x 3)
 * @param alpha      偏航惯性系数 (B x 1)，控制方向变化的平滑程度
 * @param yaw_inertia 偏航惯性参数，用于平滑旋转变化
 * 
 * @return torch::Tensor 更新后的旋转矩阵 (B x 3 x 3)
 */
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    
    // 配置CUDA执行参数：每个批次元素对应一个线程
    const int threads = a_thr.size(0);  // 批次大小
    const dim3 blocks(1);               // 单个块
    
    // 创建输出张量，与输入R具有相同的形状和设备
    torch::Tensor R_new = torch::empty_like(R);
    
    // 调用CUDA核函数更新状态向量
    AT_DISPATCH_FLOATING_TYPES(a_thr.type(), "update_state_vec", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            a_thr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            yaw_inertia);
    }));
    
    // 返回更新后的旋转矩阵
    return R_new;
}
