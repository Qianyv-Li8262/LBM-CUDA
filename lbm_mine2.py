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
ux_init = 0.1 * (term1 - term2)
ux_init = ux_init.astype(cp.float32)

f_now_3d = cp.zeros((9, totheight, totwidth), dtype=cp.float32)

usq = ux_init**2 + uy_init**2
for i in range(9):
    eu = cx[i] * ux_init + cy[i] * uy_init
    # BGK 平衡态公式
    f_now_3d[i, :, :] = w[i] * rho_init * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * usq)
f_now_gpu = f_now_3d.ravel()
f_out_gpu = f_now_gpu.copy() 
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

    mask_flat = mask_2d.flatten()
    mask_gpu = cp.array(mask_flat, dtype=cp.bool_)
    
    return mask_gpu


mask_gpu = load_mask_from_image('circle.bmp', totwidth,totheight)
base_path = os.path.dirname(os.path.abspath(__file__))

frame_count = 0

kernel_path = os.path.join(base_path, "lbm_core4.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()
module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
lbmkernel = module.get_function('fused_lbmkernel')
visualizekernel = module.get_function('visualizekernel')

stream = cp.cuda.Stream(non_blocking=True)

with stream:
    stream.begin_capture()
    lbmkernel((64,64),(32,8),(mask_gpu,f_now_gpu,f_out_gpu,ux_init,uy_init,rho_init,cp.int32(totwidth),cp.int32(totheight),cp.float32(1.8),
                                cp.float32(0.10),cp.float32(1.8)))
    lbmkernel((64,64),(32,8),(mask_gpu,f_out_gpu,f_now_gpu,ux_init,uy_init,rho_init,cp.int32(totwidth),cp.int32(totheight),cp.float32(1.8),
                                cp.float32(0.10),cp.float32(1.8)))
    graph=stream.end_capture()

for i in range(2):
    graph.launch(stream=stream)



window = zero_copy_window.ZeroCopyWindow(totwidth,totheight,'lbm')
cp.cuda.profiler.start()
iters_per_frame = 25
last_time = time.time()
while not window.should_close():
    for i in range(iters_per_frame):
        graph.launch(stream=stream)
    image = window.map_pbo()
    visualizekernel((128,32),(16,16),(ux_init,uy_init,rho_init,image,mask_gpu,cp.int32(totwidth),cp.int32(totheight),cp.float32(50.0)))
    window.unmap_and_draw()
    frame_count += 1
    # if frame_count == 2:
    #     cp.cuda.profiler.stop()
    #     sys.exit()
    if frame_count >= 10: # 每100次渲染统计一次
        duration = time.time() - last_time
        fps = frame_count / duration
        # \r 会让光标回到行首，实现原地刷新效果
        print(f"\r[LBM Simulation] FPS: {fps:.1f} | MLUPS: { (totwidth * totheight * iters_per_frame * fps*2) / 1e6 :.2f}", end="")
        last_time = time.time()
        frame_count = 0
    