/**
 * @file quadsim_kernel.cu
 * @brief 四旋翼无人机仿真的渲染和碰撞检测CUDA核函数实现
 * 
 * 本文件包含了用于无人机仿真环境的CUDA加速核函数，主要功能包括：
 * 1. 深度图渲染：从无人机视角生成深度图像，支持射线追踪
 * 2. 碰撞检测：计算无人机与环境中各种几何体的最近距离点
 * 3. 深度梯度计算：为基于深度图的学习算法提供梯度信息
 * 
 * 支持的几何体类型：
 * - 球体（balls）：三维空间中的球形障碍物
 * - 圆柱体（cylinders）：垂直方向的圆柱形障碍物
 * - 水平圆柱体（cylinders_h）：水平方向的圆柱形障碍物
 * - 立方体/长方体（voxels）：轴对齐的立方体障碍物
 * - 其他无人机：同批次中的其他无人机实例
 * 
 * 该实现采用GPU并行计算，支持批量处理多个无人机的视觉渲染，
 * 特别适用于强化学习、路径规划和自主导航算法的训练场景。
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
 * @brief CUDA核函数：无人机视角的深度图渲染
 * 
 * 从无人机的第一人称视角生成深度图像，使用射线追踪算法计算每个像素对应的最近障碍物距离。
 * 该函数支持多种几何体的碰撞检测，包括球体、圆柱体、立方体以及其他无人机。
 * 
 * @param canvas          [输出] 深度图画布 (B x H x W)，每个像素存储到最近障碍物的距离
 * @param flow            [输入] 光流信息 (B x C x H x W)，用于运动估计（当前版本未使用）
 * @param balls           [输入] 球形障碍物数组 (B x N x 4)，每个球由[x, y, z, radius]定义
 * @param cylinders       [输入] 垂直圆柱体数组 (B x N x 3)，每个圆柱由[x, y, radius]定义（高度无限）
 * @param cylinders_h     [输入] 水平圆柱体数组 (B x N x 3)，每个圆柱由[x, z, radius]定义（沿y轴延伸）
 * @param voxels          [输入] 立方体障碍物数组 (B x N x 6)，每个立方体由[cx, cy, cz, rx, ry, rz]定义
 * @param R               [输入] 当前旋转矩阵 (B x 3 x 3)，定义无人机当前姿态
 * @param R_old           [输入] 上一时刻旋转矩阵 (B x 3 x 3)，用于运动模糊（当前版本未使用）
 * @param pos             [输入] 当前位置 (B x 3)，无人机在世界坐标系中的位置
 * @param pos_old         [输入] 上一时刻位置 (B x 3)，用于运动模糊（当前版本未使用）
 * @param drone_radius    [输入] 无人机半径，用于与其他无人机的碰撞检测
 * @param n_drones_per_group [输入] 每组无人机数量，用于批量处理
 * @param fov_x_half_tan  [输入] 水平视场角的一半的正切值，用于透视投影计算
 */
template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan) {

    // 步骤1: 计算当前线程对应的像素坐标和无人机索引
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);  // 批次大小
    const int H = canvas.size(1);  // 图像高度
    const int W = canvas.size(2);  // 图像宽度
    if (c >= B * H * W) return;
    
    // 从线性索引计算三维坐标 (batch, height, width)
    const int b = c / (H * W);              // 批次索引
    const int u = (c % (H * W)) / W;        // 图像行索引（垂直）
    const int v = c % W;                    // 图像列索引（水平）
    
    // 步骤2: 计算相机投影参数和射线方向
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;  // 垂直视场角
    // 将像素坐标转换为归一化设备坐标 [-1, 1]
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    
    // 计算射线方向向量（在无人机坐标系中）
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    
    // 射线起点（无人机位置）
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // 步骤3: 初始化最小距离（与地面的交点）
    scalar_t min_dist = 100;
    scalar_t t = (-1 - oz) / dz;  // 与z=-1平面（地面）的交点
    if (t > 0) min_dist = t;

    // 步骤4: 检测与同组其他无人机的碰撞
    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;  // 跳过自己和越界的索引
        
        // 其他无人机的位置
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;  // 无人机半径
        
        // 求解射线与椭球的交点 (椭球在z方向压缩为4倍)
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;  // 判别式
        
        if (d >= 0) {  // 有交点
            r = (-b-sqrt(d)) / (2 * a);  // 近交点
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);  // 远交点
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 步骤5: 检测与球形障碍物的碰撞
    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];  // 球心x坐标
        scalar_t cy = balls[batch_base][i][1];  // 球心y坐标
        scalar_t cz = balls[batch_base][i][2];  // 球心z坐标
        scalar_t r = balls[batch_base][i][3];   // 球半径
        
        // 求解射线与球的交点
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 步骤6: 检测与垂直圆柱体的碰撞
    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];  // 圆柱中心x坐标
        scalar_t cy = cylinders[batch_base][i][1];  // 圆柱中心y坐标
        scalar_t r = cylinders[batch_base][i][2];   // 圆柱半径
        
        // 求解射线与无限高圆柱的交点（忽略z坐标）
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    
    // 步骤7: 检测与水平圆柱体的碰撞
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];  // 圆柱中心x坐标
        scalar_t cz = cylinders_h[batch_base][i][1];  // 圆柱中心z坐标
        scalar_t r = cylinders_h[batch_base][i][2];   // 圆柱半径
        
        // 求解射线与水平圆柱的交点（忽略y坐标）
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 步骤8: 检测与立方体/长方体障碍物的碰撞
    // balls
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];  // 立方体中心x坐标
        scalar_t cy = voxels[batch_base][i][1];  // 立方体中心y坐标
        scalar_t cz = voxels[batch_base][i][2];  // 立方体中心z坐标
        scalar_t rx = voxels[batch_base][i][3];  // x方向半长度
        scalar_t ry = voxels[batch_base][i][4];  // y方向半长度
        scalar_t rz = voxels[batch_base][i][5];  // z方向半长度
        
        // 使用板条法（slab method）计算射线与轴对齐立方体的交点
        scalar_t tx1 = (cx - rx - ox) / dx;  // x方向最小面
        scalar_t tx2 = (cx + rx - ox) / dx;  // x方向最大面
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        
        scalar_t ty1 = (cy - ry - oy) / dy;  // y方向最小面
        scalar_t ty2 = (cy + ry - oy) / dy;  // y方向最大面
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        
        scalar_t tz1 = (cz - rz - oz) / dz;  // z方向最小面
        scalar_t tz2 = (cz + rz - oz) / dz;  // z方向最大面
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        
        // 计算所有方向的交点范围
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        
        // 如果有有效交点，更新最小距离
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    // 步骤9: 将计算的最小距离写入输出画布
    canvas[b][u][v] = min_dist;
}

/**
 * @brief CUDA核函数：计算无人机到最近障碍物的距离和最近点
 * 
 * 计算每个无人机位置到环境中所有障碍物的最短距离，并返回最近点的坐标。
 * 该函数用于碰撞检测和避障算法，支持多种几何体类型的距离计算。
 * 
 * @param nearest_pt      [输出] 最近点坐标 (T x B x 3)，存储到最近障碍物的最近点坐标
 * @param balls           [输入] 球形障碍物数组 (B x N x 4)，每个球由[x, y, z, radius]定义
 * @param cylinders       [输入] 垂直圆柱体数组 (B x N x 3)，每个圆柱由[x, y, radius]定义
 * @param cylinders_h     [输入] 水平圆柱体数组 (B x N x 3)，每个圆柱由[x, z, radius]定义
 * @param voxels          [输入] 立方体障碍物数组 (B x N x 6)，每个立方体由[cx, cy, cz, rx, ry, rz]定义
 * @param pos             [输入] 无人机位置序列 (T x B x 3)，T为时间步数，B为无人机数量
 * @param drone_radius    [输入] 无人机半径，用于与其他无人机的距离计算
 * @param n_drones_per_group [输入] 每组无人机数量，用于批量处理
 */
template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group) {

    // 步骤1: 计算当前线程对应的时间步和无人机索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);  // 无人机数量
    const int j = idx / B;             // 时间步索引
    if (j >= nearest_pt.size(0)) return;  // 超出时间步范围
    const int b = idx % B;             // 无人机索引
    // assert(j < pos.size(0));
    // assert(b < pos.size(1));

    // 步骤2: 获取当前无人机位置
    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // 步骤3: 初始化最小距离和最近点（默认为地面）
    scalar_t min_dist = max(1e-3f, oz + 1);  // 到地面的距离（z=-1）
    scalar_t nearest_ptx = ox;               // 最近点x坐标
    scalar_t nearest_pty = oy;               // 最近点y坐标
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);  // 最近点z坐标（地面）

    // 步骤4: 计算到同组其他无人机的最近距离
    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;  // 跳过自己和越界索引
        
        // 其他无人机的位置
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;  // 无人机半径
        
        // 计算椭球距离（z方向拉伸4倍）
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);  // 表面距离
        
        if (dist < min_dist) {
            min_dist = dist;
            // 计算最近点（沿连线方向移动距离dist）
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 步骤5: 计算到球形障碍物的最近距离
    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];  // 球心坐标
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];   // 球半径
        
        // 计算到球表面的距离
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            // 最近点在球心与无人机连线上
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 步骤6: 计算到垂直圆柱体的最近距离
    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];  // 圆柱中心x坐标
        scalar_t cy = cylinders[batch_base][i][1];  // 圆柱中心y坐标
        scalar_t r = cylinders[batch_base][i][2];   // 圆柱半径
        
        // 计算到圆柱面的距离（只考虑xy平面）
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            // 最近点保持z坐标不变
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz;
        }
    }
    
    // 步骤7: 计算到水平圆柱体的最近距离
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];  // 圆柱中心x坐标
        scalar_t cz = cylinders_h[batch_base][i][1];  // 圆柱中心z坐标
        scalar_t r = cylinders_h[batch_base][i][2];   // 圆柱半径
        
        // 计算到圆柱面的距离（只考虑xz平面）
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            // 最近点保持y坐标不变
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy;
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 步骤8: 计算到立方体/长方体的最近距离
    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];  // 立方体中心坐标
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        
        // 计算最大距离以确保在立方体外部
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);  // x方向半长度
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);  // y方向半长度
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);  // z方向半长度
        
        // 计算立方体表面的最近点
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));  // 限制在立方体范围内
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        
        // 计算到最近点的距离
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    
    // 步骤9: 将最近点坐标写入输出
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


/**
 * @brief CUDA核函数：计算深度图的法向量梯度
 * 
 * 根据深度图计算每个像素的法向量，用于基于深度图的视觉算法和梯度计算。
 * 该函数通过分析相邻像素的深度差异来估计表面法向量，为深度学习算法提供几何信息。
 * 
 * @param depth           [输入] 高分辨率深度图 (B x 1 x H*2 x W*2)，原始深度数据
 * @param dddp            [输出] 法向量梯度 (B x 3 x H x W)，每个像素的单位法向量
 *                               dddp[b][0][u][v] = -1/norm (向前方向分量)
 *                               dddp[b][1][u][v] = dy/norm (垂直方向分量)  
 *                               dddp[b][2][u][v] = dz/norm (水平方向分量)
 * @param fov_x_half_tan  [输入] 水平视场角的一半的正切值，用于透视校正
 */
template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    // 步骤1: 计算当前线程对应的像素坐标
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);  // 批次大小
    const int H = dddp.size(2);  // 输出图像高度
    const int W = dddp.size(3);  // 输出图像宽度
    if (c >= B * H * W) return;
    
    // 从线性索引计算三维坐标
    const int b = c / (H * W);              // 批次索引
    const int u = (c % (H * W)) / W;        // 行索引
    const int v = c % W;                    // 列索引

    // 步骤2: 从高分辨率深度图采样2x2区域并计算平均深度
    const scalar_t unit = fov_x_half_tan / W;  // 每个像素对应的角度单位
    // 采样2x2区域的深度值并计算平均值
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + 
                       depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    
    // 步骤3: 计算深度梯度（有限差分法）
    // 垂直方向梯度（上下像素的深度差）
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - 
                          depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    
    // 水平方向梯度（左右像素的深度差）
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + 
                          depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    
    // 步骤4: 计算法向量并归一化
    // 对应的Python代码：
    // if ReRender.diff_kernel is None:
    //     unit = 0.637 / depth.size(3)
    //     ReRender.diff_kernel = torch.tensor([
    //         [[1, -1], [1, -1]],    // 水平差分核
    //         [[1, 1], [-1, -1]],    // 垂直差分核
    //         [[unit, unit], [unit, unit]],  // 深度缩放核
    //     ], device=device).mul(0.5)[:, None]
    // ddepthdyz = F.conv2d(depth, ReRender.diff_kernel, None, 2)
    // depth = ddepthdyz[:, 2:]
    // ddepthdyz = torch.cat([
    //     torch.full_like(depth, -1.),
    //     ddepthdyz[:, :2] / depth,
    // ], 1)
    
    // 计算法向量的模长，最小值限制为8（避免过大的梯度）
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    
    // 归一化后的法向量分量
    dddp[b][0][u][v] = -1. / dddp_norm;      // x分量（向前方向）
    dddp[b][1][u][v] = dddy / dddp_norm;     // y分量（垂直方向）
    dddp[b][2][u][v] = dddz / dddp_norm;     // z分量（水平方向）
    
    // 对应的Python代码：
    // ddepthdyz /= ddepthdyz.norm(2, 1, True).clamp_min(8);
}

} // namespace

/**
 * @brief C++接口函数：无人机视角深度图渲染
 * 
 * 这是一个C++到CUDA的接口函数，用于从Python/PyTorch调用CUDA核函数进行深度图渲染。
 * 该函数使用射线追踪算法生成无人机第一人称视角的深度图像。
 * 
 * @param canvas          深度图输出画布 (B x H x W)
 * @param flow            光流信息 (B x C x H x W)，当前版本未使用
 * @param balls           球形障碍物数组 (B x N x 4)
 * @param cylinders       垂直圆柱体数组 (B x N x 3)
 * @param cylinders_h     水平圆柱体数组 (B x N x 3)
 * @param voxels          立方体障碍物数组 (B x N x 6)
 * @param R               当前旋转矩阵 (B x 3 x 3)
 * @param R_old           上一时刻旋转矩阵 (B x 3 x 3)，当前版本未使用
 * @param pos             当前位置 (B x 3)
 * @param pos_old         上一时刻位置 (B x 3)，当前版本未使用
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
    float fov_x_half_tan) {
    
    // 配置CUDA执行参数
    const int threads = 1024;                          // 每个块的线程数
    size_t state_size = canvas.numel();                // 总像素数量
    const dim3 blocks((state_size + threads - 1) / threads);  // 所需块数

    // 调用CUDA核函数进行深度图渲染
    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
    }));
}

/**
 * @brief C++接口函数：深度图法向量计算
 * 
 * 这是一个C++到CUDA的接口函数，用于计算深度图的法向量梯度信息。
 * 该函数通过分析深度图的局部变化来估计表面法向量，为基于深度的视觉算法提供几何特征。
 * 
 * @param depth           高分辨率深度图 (B x 1 x H*2 x W*2)
 * @param dddp            法向量梯度输出 (B x 3 x H x W)
 * @param fov_x_half_tan  水平视场角一半的正切值
 */
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    
    // 配置CUDA执行参数
    const int threads = 1024;                          // 每个块的线程数
    size_t state_size = dddp.numel();                  // 总像素数量
    const dim3 blocks((state_size + threads - 1) / threads);  // 所需块数

    // 调用CUDA核函数计算法向量梯度
    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

/**
 * @brief C++接口函数：查找最近障碍物点
 * 
 * 这是一个C++到CUDA的接口函数，用于计算无人机位置到环境中最近障碍物的距离和最近点坐标。
 * 该函数支持多种几何体的距离计算，广泛用于碰撞检测、避障规划和安全约束等应用。
 * 
 * @param nearest_pt      最近点输出 (T x B x 3)，存储到最近障碍物的最近点坐标
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
    int n_drones_per_group) {
    
    // 配置CUDA执行参数
    const int threads = 1024;                          // 每个块的线程数
    size_t state_size = pos.size(0) * pos.size(1);     // 总计算数量 (时间步 × 无人机数)
    const dim3 blocks((state_size + threads - 1) / threads);  // 所需块数
    
    // 调用CUDA核函数查找最近点
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
}
