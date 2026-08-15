import cupy as cp
import numpy as np
import zero_copy_window
import cv2
import os
import time,sys

totwidth = 4096
totheight = 1024
totpixels = totwidth * totheight

w = cp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=cp.float32)
cx = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=cp.float32)
cy = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=cp.float32)

x = cp.arange(totwidth, dtype=cp.float32)
y = cp.arange(totheight, dtype=cp.float32)
X, Y = cp.meshgrid(x, y)

rho_init = cp.ones((totheight, totwidth), dtype=cp.float32)
uy_init = cp.zeros((totheight, totwidth), dtype=cp.float32)
term1 = cp.exp(-((X - 150)**2 + (Y - 150)**2) / (2 * 10**2))
term2 = cp.exp(-((X - 300)**2 + (Y - 150)**2) / (2 * 10**2))
ux_init = (0.1 * (term1 - term2)).astype(cp.float32)

f_now_3d = cp.zeros((9, totheight, totwidth), dtype=cp.float32)
usq = ux_init**2 + uy_init**2
for i in range(9):
    eu = cx[i] * ux_init + cy[i] * uy_init
    f_now_3d[i] = w[i] * rho_init * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * usq)

f_now_gpu = f_now_3d.ravel()
f_out_gpu = cp.empty_like(f_now_gpu)
print("Fluid field initialized successfully!")

def load_mask_from_image(file_path, totwidth, totheight):
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"无法找到图片: {file_path}")
    img = cv2.resize(img, (totwidth, totheight))
    mask_2d = (img[:, :, 1] == 0).astype(np.bool_)
    mask_2d[0, :] = True
    mask_2d[totheight - 1, :] = True
    mask_2d[1:totheight-1, 0] = False
    mask_2d[1:totheight-1, totwidth-1] = False
    return cp.array(mask_2d.ravel(), dtype=cp.bool_)

mask_gpu = load_mask_from_image("circle.bmp", totwidth, totheight)

base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "lbm_core_smemaa_endfused.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=("-use_fast_math",))
lbmkernel = module.get_function("fused_lbmkernel")
macrokernel = module.get_function("macro_kernel")
visualizekernel = module.get_function("visualizekernel")

SQUARE_A = 24
T = 6
block = (32, 16)
grid = ((totwidth + SQUARE_A - 1) // SQUARE_A, (totheight + SQUARE_A - 1) // SQUARE_A)
vis_block = (16, 16)
vis_grid = ((totwidth + vis_block[0] - 1) // vis_block[0], (totheight + vis_block[1] - 1) // vis_block[1])

tau_inv = cp.float32(1.0)
u_in = cp.float32(0.10)
width_gpu = cp.int32(totwidth)
height_gpu = cp.int32(totheight)

args_now_out = (mask_gpu, f_now_gpu, f_out_gpu, width_gpu, height_gpu, tau_inv, u_in)
args_out_now = (mask_gpu, f_out_gpu, f_now_gpu, width_gpu, height_gpu, tau_inv, u_in)

stream = cp.cuda.Stream(non_blocking=True)
with stream:
    stream.begin_capture()
    lbmkernel(grid, block, args_now_out)
    lbmkernel(grid, block, args_out_now)
    graph = stream.end_capture()

for _ in range(10):
    graph.launch(stream=stream)
stream.synchronize()

window = zero_copy_window.ZeroCopyWindow(totwidth, totheight, "lbm")
cp.cuda.profiler.start()

iters_per_frame = 50
frame_count = 0
last_time = time.time()

while not window.should_close():
    for _ in range(iters_per_frame):
        graph.launch(stream=stream)
    stream.synchronize()

    macrokernel(vis_grid, vis_block, (f_now_gpu, ux_init, uy_init, rho_init, mask_gpu, width_gpu, height_gpu, u_in))

    image = window.map_pbo()
    visualizekernel(vis_grid, vis_block, (ux_init, uy_init, rho_init, image, mask_gpu, width_gpu, height_gpu, cp.float32(50.0)))
    window.unmap_and_draw()

    frame_count += 1
    # if frame_count == 2:
    #     cp.cuda.profiler.stop()
    #     sys.exit()
    if frame_count >= 10:
        cp.cuda.Device().synchronize()
        duration = time.time() - last_time
        fps = frame_count / duration
        mlups = totpixels * iters_per_frame * (2 * T) * fps / 1e6
        print(f"\r[LBM Simulation] FPS: {fps:.1f} | MLUPS: {mlups:.7f}", end="")
        last_time = time.time()
        frame_count = 0
