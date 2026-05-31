import numpy as np
import cupy as cp
import ctypes
import glfw
from OpenGL.GL import *
import os
import glob

# ============ 找到 cudart DLL ============
def load_cudart():
    cupy_cuda_path = os.path.join(os.path.dirname(cp.__file__), 'cuda')
    dll_candidates = glob.glob(os.path.join(cupy_cuda_path, 'bin', 'cudart64_*.dll'))
    if not dll_candidates:
        dll_candidates = glob.glob(os.path.join(cupy_cuda_path, '..', '**', 'cudart64_*.dll'), recursive=True)
    if not dll_candidates:
        # fallback: search in PATH
        for p in os.environ.get('PATH','').split(';'):
            dll_candidates += glob.glob(os.path.join(p, 'cudart64_*.dll'))
    if dll_candidates:
        cudart = ctypes.cdll.LoadLibrary(dll_candidates[0])
        print(f'Loaded cudart: {dll_candidates[0]}')
        return cudart
    else:
        # last resort: try cupy's internal handle
        cudart = ctypes.CDLL('cudart64_12.dll')
        print('Loaded cudart64_12.dll from system PATH')
        return cudart

cudart = load_cudart()

# ============ CUDA-GL interop 辅助函数 ============
cudaGraphicsRegisterFlagsWriteDiscard = 2

class cudaGraphicsResource_p(ctypes.c_void_p):
    pass

def cuda_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {msg}")

def register_gl_buffer(gl_buffer_id):
    resource = cudaGraphicsResource_p()
    err = cudart.cudaGraphicsGLRegisterBuffer(
        ctypes.byref(resource),
        ctypes.c_uint(gl_buffer_id),
        ctypes.c_uint(cudaGraphicsRegisterFlagsWriteDiscard))
    cuda_check(err, "cudaGraphicsGLRegisterBuffer")
    return resource

def map_resource(resource):
    err = cudart.cudaGraphicsMapResources(ctypes.c_int(1), ctypes.byref(resource), ctypes.c_void_p(0))
    cuda_check(err, "cudaGraphicsMapResources")

def get_mapped_pointer(resource):
    dev_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    err = cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(size), resource)
    cuda_check(err, "cudaGraphicsResourceGetMappedPointer")
    return dev_ptr.value, size.value

def unmap_resource(resource):
    err = cudart.cudaGraphicsUnmapResources(ctypes.c_int(1), ctypes.byref(resource), ctypes.c_void_p(0))
    cuda_check(err, "cudaGraphicsUnmapResources")

class ZeroCopyWindow:
    def __init__(self, width, height, title="Zero-Copy Window"):
        self.width = width
        self.height = height
        self.title = title
        
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
            
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # 不限帧率

        # 创建 OpenGL PBO
        self.pbo_size = width * height * 4  # RGBA uint8
        self.pbo_id = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self.pbo_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # 创建 OpenGL 纹理
        self.gl_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gl_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        # 注册 PBO 到 CUDA
        self.cuda_pbo_resource = register_gl_buffer(int(self.pbo_id))
        
        # 键盘状态
        self.key_pressed = {}
        glfw.set_key_callback(self.window, self._key_callback)
        
        print("OpenGL + CUDA interop initialized. Zero-copy ready!")

    def _key_callback(self, win, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.key_pressed[key] = True
        elif action == glfw.RELEASE:
            self.key_pressed.pop(key, None)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def map_pbo(self):
        """映射 PBO 到 CuPy 数组，返回一个 1D uint8 的 CuPy 数组"""
        map_resource(self.cuda_pbo_resource)
        pbo_dev_ptr, pbo_dev_size = get_mapped_pointer(self.cuda_pbo_resource)

        # 用 CuPy 包装 PBO 的 GPU 指针 (无拷贝)
        pbo_mem = cp.cuda.UnownedMemory(pbo_dev_ptr, pbo_dev_size, owner=None)
        pbo_memptr = cp.cuda.MemoryPointer(pbo_mem, 0)
        pbo_cupy = cp.ndarray(pbo_dev_size, dtype=cp.uint8, memptr=pbo_memptr)
        
        return pbo_cupy

    def unmap_and_draw(self):
        """解除映射并将 PBO 内容绘制到屏幕"""
        unmap_resource(self.cuda_pbo_resource)

        # --- OpenGL: PBO → 纹理 → 全屏四边形 ---
        glClear(GL_COLOR_BUFFER_BIT)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glBindTexture(GL_TEXTURE_2D, self.gl_tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0,1); glVertex2f(-1,-1)
        glTexCoord2f(1,1); glVertex2f(1,-1)
        glTexCoord2f(1,0); glVertex2f(1,1)
        glTexCoord2f(0,0); glVertex2f(-1,1)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def set_title(self, title):
        glfw.set_window_title(self.window, title)

    def destroy(self):
        glfw.terminate()
