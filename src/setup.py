"""
setup.py - 四旋翼无人机仿真CUDA扩展模块构建脚本

该脚本用于编译和安装 quadsim_cuda 扩展模块，该模块是一个基于CUDA的
四旋翼无人机物理仿真引擎，用于深度强化学习训练和仿真。

主要功能：
- 编译C++/CUDA源代码为Python可调用的扩展模块
- 配置CUDA编译选项和依赖项
- 集成PyTorch的CUDA扩展构建系统

使用方法：
    python setup.py build_ext --inplace  # 就地编译
    python setup.py install              # 安装到Python环境

依赖项：
    - PyTorch (提供CUDA扩展支持)
    - CUDA开发工具包
    - C++编译器 (如Visual Studio on Windows)
"""

# 导入必要的构建工具
from setuptools import setup                        # Python包构建基础工具
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # PyTorch CUDA扩展支持

# 配置CUDA扩展模块
setup(
    # 模块名称，安装后可通过 import quadsim_cuda 导入
    name='quadsim_cuda',
    
    # 扩展模块定义
    ext_modules=[
        # 创建CUDA扩展模块
        # 参数说明：
        #   第一个参数: 模块名称，对应Python中的导入名
        #   第二个参数: 源代码文件列表
        CUDAExtension('quadsim_cuda', [
            'quadsim.cpp',          # C++接口文件，提供Python绑定
            'quadsim_kernel.cu',    # CUDA核函数，实现四旋翼仿真逻辑
            'dynamics_kernel.cu',   # CUDA核函数，实现动力学计算
        ]),
    ],
    
    # 自定义构建命令
    # 使用PyTorch提供的BuildExtension来处理CUDA编译
    cmdclass={
        'build_ext': BuildExtension  # 扩展构建命令，自动处理CUDA编译选项
    })
