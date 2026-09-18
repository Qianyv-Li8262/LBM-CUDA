# LBM CUDA 性能优化实验

这是一个基于 D2Q9（二维九速度）格子玻尔兹曼方法（LBM）的流体模拟实验项目。项目从易于验证的 CPU 原型出发，逐步迁移到 GPU，并通过时域分块（temporal blocking）提升显存数据复用效率。

## 技术路线与性能演进

| 阶段 | 实现思路 | 吞吐量 |
| --- | --- | ---: |
| 1 | Naive CPU / Taichi 原型 | 100 MLUPS/s |
| 2 | Naive SoA GPU CUDA 实现 | 4,000 MLUPS/s |
| 3 | Temporal blocking CUDA 实现 | 17,000 MLUPS/s |

MLUPS/s 表示每秒更新的百万格点（Million Lattice Updates Per Second）。性能会随 GPU 型号、网格尺寸、编译选项及测试场景变化。

## 优化要点

- **CPU 原型**：使用 Taichi 快速验证 LBM 碰撞与迁移逻辑。
- **SoA 数据布局**：将九个分布函数按方向分离存储，便于 CUDA 线程连续访问数据。
- **CUDA 核函数融合**：在同一核函数内完成主要 LBM 更新步骤，减少全局内存往返。
- **Temporal blocking**：每次加载数据后在片上存储中连续推进多个时间步，以提高数据复用并降低全局内存带宽压力。

## 主要文件

- `lbm.py`：Taichi 实现的原型。
- `lbm_smemaa_main.py`：使用 CuPy 编译、调用 CUDA 核函数的入口。
- `lbm_core_smemaa.cu`：naive CUDA 实现。
- `lbm_core_smemaa_endfused.cu`：temporal blocking CUDA 实现。最快。
- `lbm_core_smemaa_policy.cu`：其他优化策略与实验版本。
- `circle.bmp`：示例障碍物掩码。

## 依赖

Python 环境需要安装 `numpy`、`opencv-python`、`taichi` 和与本机 CUDA 匹配的 `cupy`。运行 CUDA 版本时需要可用的 NVIDIA GPU 及 CUDA 驱动。
