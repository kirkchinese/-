/**
 * @file quadsim.cpp
 * @brief 四旋翼无人机仿真系统的Python C++扩展接口
 * 
 * 本文件作为Python和CUDA核函数之间的桥梁，提供PyTorch C++扩展接口，
 * 将无人机动力学仿真和视觉渲染的CUDA实现暴露给Python环境。
 * 
 * 主要功能包括：
 * 1. 无人机动力学仿真：前向传播、反向传播、状态更新
 * 2. 视觉渲染：深度图生成、法向量计算
 * 3. 碰撞检测：最近点查找、距离计算
 * 
 * 该扩展模块使得Python代码能够高效调用GPU加速的仿真核函数，
 * 特别适用于强化学习、深度学习和大规模并行仿真应用。
 * 
 * 编译后的模块可以在Python中直接导入使用：
 * import quadsim_cuda
 * 
 * @author DiffPhys-UAV项目组
 * @date 2025年9月21日
 */

#include <torch/extension.h>

#include <vector>

//=============================================================================
// CUDA核函数声明部分
// 这些函数的具体实现在对应的.cu文件中，此处仅提供声明以便C++调用
//=============================================================================

/**
 * @brief 无人机视角深度图渲染CUDA函数声明
 * 
 * 从无人机第一人称视角生成深度图像，支持多种几何体的射线追踪渲染。
 * 
 * @param canvas          深度图输出画布 (B x H x W)
 * @param flow            光流信息 (B x C x H x W)，当前版本未使用
 * @param balls           球形障碍物数组 (B x N x 4)
 * @param cylinders       垂直圆柱体数组 (B x N x 3)
 * @param cylinders_h     水平圆柱体数组 (B x N x 3)
 * @param voxels          立方体障碍物数组 (B x N x 6)
 * @param R               当前旋转矩阵 (B x 3 x 3)
 * @param R_old           上一时刻旋转矩阵 (B x 3 x 3)
 * @param pos             当前位置 (B x 3)
 * @param pos_old         上一时刻位置 (B x 3)
 * @param drone_radius    无人机半径
 * @param n_drones_per_group 每组无人机数量
 * @param fov_x_half_tan  水平视场角一半的正切值
 */
void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

/**
 * @brief 深度图法向量计算CUDA函数声明
 * 
 * 根据深度图计算表面法向量，为基于深度的视觉算法提供几何特征。
 * 
 * @param depth           高分辨率深度图 (B x 1 x H*2 x W*2)
 * @param dddp            法向量梯度输出 (B x 3 x H x W)
 * @param fov_x_half_tan  水平视场角一半的正切值
 */
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

/**
 * @brief 最近障碍物点查找CUDA函数声明
 * 
 * 计算无人机位置到环境中最近障碍物的距离和最近点坐标。
 * 
 * @param nearest_pt      最近点输出 (T x B x 3)
 * @param balls           球形障碍物数组 (B x N x 4)
 * @param cylinders       垂直圆柱体数组 (B x N x 3)
 * @param cylinders_h     水平圆柱体数组 (B x N x 3)
 * @param voxels          立方体障碍物数组 (B x N x 6)
 * @param pos             无人机位置序列 (T x B x 3)
 * @param drone_radius    无人机半径
 * @param n_drones_per_group 每组无人机数量
 */
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

/**
 * @brief 无人机状态向量更新CUDA函数声明
 * 
 * 根据推力向量和预测速度更新无人机的姿态旋转矩阵。
 * 
 * @param R          当前旋转矩阵 (B x 3 x 3)
 * @param a_thr      推力加速度向量 (B x 3)
 * @param v_pred     预测速度向量 (B x 3)
 * @param alpha      偏航惯性系数 (B x 1)
 * @param yaw_inertia 偏航惯性参数
 * @return torch::Tensor 更新后的旋转矩阵 (B x 3 x 3)
 */
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

/**
 * @brief 无人机动力学前向传播CUDA函数声明
 * 
 * 根据当前状态和控制输入计算下一时刻的无人机状态。
 * 
 * @param R               当前旋转矩阵 (B x 3 x 3)
 * @param dg              重力扰动向量 (B x 3)
 * @param z_drag_coef     垂直阻力系数 (B x 1)
 * @param drag_2          阻力系数 (B x 2)
 * @param pitch_ctl_delay 俯仰控制延迟 (B x 1)
 * @param act_pred        预测执行器输出 (B x 3)
 * @param act             当前执行器状态 (B x 3)
 * @param p               当前位置 (B x 3)
 * @param v               当前速度 (B x 3)
 * @param v_wind          风速向量 (B x 3)
 * @param a               当前加速度 (B x 3)
 * @param ctl_dt          控制时间步长
 * @param airmode_av2a    空中模式转换系数
 * @return std::vector<torch::Tensor> [act_next, p_next, v_next, a_next]
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
    float airmode_av2a);

/**
 * @brief 无人机动力学反向传播CUDA函数声明
 * 
 * 计算前向传播过程的梯度，用于基于梯度的优化算法。
 * 
 * @param R               旋转矩阵 (B x 3 x 3)
 * @param dg              重力扰动 (B x 3)
 * @param z_drag_coef     垂直阻力系数 (B x 1)
 * @param drag_2          阻力系数 (B x 2)
 * @param pitch_ctl_delay 俯仰控制延迟 (B x 1)
 * @param v               速度 (B x 3)
 * @param v_wind          风速 (B x 3)
 * @param act_next        下一时刻执行器状态 (B x 3)
 * @param _d_act_next     执行器状态梯度 (B x 3)
 * @param d_p_next        位置梯度 (B x 3)
 * @param d_v_next        速度梯度 (B x 3)
 * @param _d_a_next       加速度梯度 (B x 3)
 * @param grad_decay      梯度衰减系数
 * @param ctl_dt          控制时间步长
 * @return std::vector<torch::Tensor> [d_act_pred, d_act, d_p, d_v, d_a]
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
    float ctl_dt);

//=============================================================================
// C++接口部分
//=============================================================================

/**
 * 以下是早期版本的输入验证宏定义和包装函数示例，当前版本已不使用。
 * 这些代码展示了如何添加CUDA张量和连续性检查，可作为参考。
 * 
 * 在PyTorch的较新版本中，张量验证通常在Python端进行，
 * 或者使用更现代的错误处理机制。
 */

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

//=============================================================================
// Python模块绑定
// 使用pybind11将C++函数暴露给Python环境
//=============================================================================

/**
 * @brief Python扩展模块定义
 * 
 * 使用pybind11库将CUDA函数绑定到Python模块中，使得Python代码能够
 * 直接调用这些高性能的GPU加速函数。
 * 
 * 绑定的函数包括：
 * - render: 深度图渲染
 * - find_nearest_pt: 最近点查找  
 * - update_state_vec: 状态向量更新
 * - run_forward: 动力学前向传播
 * - run_backward: 动力学反向传播
 * - rerender_backward: 深度图法向量计算
 * 
 * Python中的使用示例：
 * import quadsim_cuda
 * depth_map = quadsim_cuda.render(canvas, flow, balls, ...)
 * gradients = quadsim_cuda.run_backward(R, dg, ...)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 深度图渲染接口
  m.def("render", &render_cuda, "render (CUDA)");
  
  // 最近点查找接口  
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  
  // 状态向量更新接口
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  
  // 动力学前向传播接口
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  
  // 动力学反向传播接口
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  
  // 深度图法向量计算接口
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
}
