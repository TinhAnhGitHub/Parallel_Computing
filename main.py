import numpy as np
import ctypes
import time
import argparse
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

def ensure_context():
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(1, 1, "compute", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(win)
    return win

def create_ssbo_from_numpy(binding, np_array, usage=GL_STATIC_DRAW):
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    size = np_array.nbytes
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, None, usage)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, ctypes.c_void_p(np_array.ctypes.data))
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo

def create_ssbo_reserve(binding, size_bytes, usage=GL_DYNAMIC_READ):
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, None, usage)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo

def read_ssbo_to_numpy(ssbo, nbytes, dtype=np.float32, count=None):
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    buf = ctypes.create_string_buffer(nbytes)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nbytes, buf)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    if count is None:
        count = nbytes // np.dtype(dtype).itemsize
    arr = np.frombuffer(buf, dtype=dtype, count=count).copy()
    return arr

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["naive", "chunked"], default="naive")
parser.add_argument("--size", choices=["small", "medium", "big"], default="medium")
args = parser.parse_args()

mode = args.mode
N = 0
if args.size=="small":
    N=128
elif args.size=="medium":
    N=1024
elif args.size=="big":
    N=8192
TILE = 16
BLOCK_ROWS = 512  

shader_file = f"matmul_{mode}.comp"
with open(shader_file, "r") as f:
    src = f.read()

win = ensure_context()
prog = compileProgram(compileShader(src, GL_COMPUTE_SHADER))
glUseProgram(prog)

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.empty((N, N), dtype=np.float32)

if mode == "chunked":
    Npad = N if N % TILE != 0 else N + TILE
    Apad = np.zeros((Npad, Npad), dtype=np.float32)
    Bpad = np.zeros((Npad, Npad), dtype=np.float32)
    Apad[:N,:N] = A
    Bpad[:N,:N] = B
    Cpad = np.empty((Npad, Npad), dtype=np.float32)
else:
    Npad = N
    Apad = A
    Bpad = B
    Cpad = C

ssbo_A = create_ssbo_from_numpy(0, Apad)
ssbo_B = create_ssbo_from_numpy(1, Bpad)
ssbo_C = create_ssbo_reserve(2, Cpad.nbytes)

locN = glGetUniformLocation(prog, "N")
if locN != -1: glUniform1i(locN, N)
locStride = glGetUniformLocation(prog, "stride")
if locStride != -1: glUniform1i(locStride, Npad)
locBase = glGetUniformLocation(prog, "baseRow")  


start_time = time.time()

if mode == "naive":
    groups_x = (N + TILE - 1) // TILE
    groups_y = (N + TILE - 1) // TILE
    glDispatchCompute(groups_x, groups_y, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    glFinish()
else:  
    groups_x = (Npad + TILE - 1) // TILE
    for base in range(0, N, BLOCK_ROWS):
        rows_this = min(BLOCK_ROWS, N - base)
        groups_y = (rows_this + TILE - 1) // TILE
        glUseProgram(prog)
        glUniform1i(locBase, base)
        glDispatchCompute(groups_x, groups_y, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    glFinish()

end_time = time.time()
gpu_time = (end_time - start_time)*1000.0
print(f"Matrix size: {N}x{N}, shader={mode}")
print(f"GPU walltime = {gpu_time:.3f} ms")

arr = read_ssbo_to_numpy(ssbo_C, Cpad.nbytes, count=Npad*Npad)
Cpad_gpu = arr.reshape((Npad, Npad))
C_gpu = Cpad_gpu[:N, :N]   


samples = [(0,0),(N//2,N//3),(N-1,N-1)]
for i,j in samples:
    gpu_val = float(C_gpu[i,j])
    cpu_val = float(np.dot(A[i,:], B[:,j]))
    print(f"sample ({i},{j}): GPU={gpu_val:.6f}, CPU={cpu_val:.6f}, diff={abs(gpu_val-cpu_val):.6f}")


glDeleteBuffers(3, (GLuint*3)(ssbo_A, ssbo_B, ssbo_C))
glDeleteProgram(prog)
glfw.terminate()
