import cupy as cp
import numpy as np
import zero_copy_window
import cv2
import os
import time
import sys
totwidth = 2048
totheight = 512
totpixels = totwidth * totheight

# D2Q9
w = cp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=cp.float32)
cx = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=cp.float32)
cy = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=cp.float32)

# ============================================================
# Initial condition
# ============================================================

x = cp.arange(totwidth, dtype=cp.float32)
y = cp.arange(totheight, dtype=cp.float32)
X, Y = cp.meshgrid(x, y)

rho_init = cp.ones((totheight, totwidth), dtype=cp.float32)
uy_init = cp.zeros((totheight, totwidth), dtype=cp.float32)

term1 = cp.exp(-((X - 150)**2 + (Y - 150)**2) / (2 * 10**2))
term2 = cp.exp(-((X - 300)**2 + (Y - 150)**2) / (2 * 10**2))
ux_init = (0.1 * (term1 - term2)).astype(cp.float32)

# 调试静止场时：
# ux_init[:] = 0

f_now_3d = cp.zeros((9, totheight, totwidth), dtype=cp.float32)
usq = ux_init**2 + uy_init**2

for i in range(9):
    eu = cx[i] * ux_init + cy[i] * uy_init
    f_now_3d[i] = w[i] * rho_init * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * usq)

# global macro-step ping-pong
f_now_gpu = f_now_3d.ravel()
f_out_gpu = cp.empty_like(f_now_gpu)

print("Fluid field initialized successfully!")


# ============================================================
# Mask
# ============================================================

def load_mask_from_image(file_path, totwidth, totheight):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"无法找到图片: {file_path}")

    img = cv2.resize(img, (totwidth, totheight))
    mask_2d = (img[:, :, 1] == 0).astype(np.bool_)

    mask_2d[0, :] = True
    mask_2d[totheight - 1, :] = True

    # inlet / outlet
    mask_2d[1:totheight-1, 0] = False
    mask_2d[1:totheight-1, totwidth-1] = False

    return cp.array(mask_2d.ravel(), dtype=cp.bool_)


mask_gpu = load_mask_from_image("circle.bmp", totwidth, totheight)


# ============================================================
# Compile CUDA
# ============================================================

base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "lbm_core_smemaa.cu")

with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=("-use_fast_math",))
lbmkernel = module.get_function("fused_lbmkernel")
visualizekernel = module.get_function("visualizekernel")


# ============================================================
# Launch configuration
# ============================================================

# Must match constexpr values in lbm_core_smemaa.cu
SQUARE_A = 24
T = 4

block = (32, 8)
grid = ((totwidth + SQUARE_A - 1) // SQUARE_A,
        (totheight + SQUARE_A - 1) // SQUARE_A)

vis_block = (16, 16)
vis_grid = ((totwidth + vis_block[0] - 1) // vis_block[0],
            (totheight + vis_block[1] - 1) // vis_block[1])

print(f"LBM grid={grid}, block={block}, core={SQUARE_A}, T={T}, smem tile={SQUARE_A + 2*T}")

tau_inv = cp.float32(1.7)
u_in = cp.float32(0.10)

width_gpu = cp.int32(totwidth)
height_gpu = cp.int32(totheight)

args_now_out = (
    mask_gpu, f_now_gpu, f_out_gpu,
    ux_init, uy_init, rho_init,
    width_gpu, height_gpu, tau_inv, u_in
)

args_out_now = (
    mask_gpu, f_out_gpu, f_now_gpu,
    ux_init, uy_init, rho_init,
    width_gpu, height_gpu, tau_inv, u_in
)


# ============================================================
# CUDA Graph
#
# one graph:
#
# f_now --6 steps--> f_out --6 steps--> f_now
#
# = 12 physical LBM timesteps
# ============================================================

stream = cp.cuda.Stream(non_blocking=True)

with stream:
    stream.begin_capture()

    lbmkernel(grid, block, args_now_out)
    lbmkernel(grid, block, args_out_now)

    graph = stream.end_capture()


# graph warmup
for _ in range(2):
    graph.launch(stream=stream)

stream.synchronize()


# ============================================================
# Window
# ============================================================

window = zero_copy_window.ZeroCopyWindow(totwidth, totheight, "lbm")
cp.cuda.profiler.start()

# 1 graph = 2 macro kernels = 12 LBM timesteps.
#
# 原来 iters_per_frame=8 个单 kernel 是 48 steps/frame，
# 所以这里用 4 张 graph 仍然是 48 steps/frame。
iters_per_frame = 50

frame_count = 0
last_time = time.time()

while not window.should_close():

    # Simulation
    for _ in range(iters_per_frame):
        graph.launch(stream=stream)

    # graph 在 non-blocking stream，显示前必须确保模拟完成。
    stream.synchronize()

    # Visualization
    image = window.map_pbo()

    visualizekernel(
        (128, 32), (16, 16),
        (ux_init, uy_init, rho_init, image, mask_gpu,
         width_gpu, height_gpu, cp.float32(50.0))
    )

    window.unmap_and_draw()

    # Performance
    frame_count += 1
    # if frame_count == 2:
    #     cp.cuda.profiler.stop()
    #     sys.exit()
    if frame_count >= 10:
        cp.cuda.Device().synchronize()

        duration = time.time() - last_time
        fps = frame_count / duration

        # one graph = 2 kernels * 6 steps/kernel
        mlups = totpixels * iters_per_frame * (2 * T) * fps / 1e6

        print(f"\r[LBM Simulation] FPS: {fps:.1f} | MLUPS: {mlups:.2f}", end="")

        last_time = time.time()
        frame_count = 0