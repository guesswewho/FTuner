import tvm
from tvm import auto_scheduler, te, topi

import logging
import numpy as np

logger = logging.getLogger(__name__)

from ...shared import CUDATarget, CUDAContext


@auto_scheduler.register_workload
def BatchMatmulNT(B, M, K, N):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, N, K), name='W')
    Y = topi.nn.batch_matmul(X, W, transpose_b=True)
    return [X, W, Y]


@auto_scheduler.register_workload
def BatchMatmulNN(B, M, K, N):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, K, N), name='W')
    Y = topi.nn.batch_matmul(X, W, transpose_b=False)
    return [X, W, Y]

@auto_scheduler.register_workload
def ConvNCHW(N, C, H, W, K, R, S, groups=1, conv_dtype='float32'):
    X = te.placeholder((N, C, H, W), name='X', dtype=conv_dtype)
    W = te.placeholder((K, C//groups, R, S), name='W', dtype=conv_dtype)
    pad = [R//2, S//2]
    stride = [1, 1]
    dilation = [0, 0]
    Y = topi.nn.conv2d(X, W, stride, pad, dilation, out_dtype=conv_dtype)
    return [X, W, Y]

def _vendor_batch_matmul_kernel(B, M, K, N, transpose_b):
    X = te.placeholder((B, M, K), name='X')
    W = te.placeholder((B, N, K) if transpose_b else (B, K, N), name='W')
    Y = tvm.contrib.cublas.batch_matmul(X, W, transa=False, transb=transpose_b)
    sched = te.create_schedule(Y.op)
    vendor_kernel = tvm.build(sched, [X, W, Y], CUDATarget)
    return vendor_kernel, Y

def _vendor_conv_kernel(N, C, H, W, K, R, S, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, groups=1):
    if tensor_format == 0:
        X = te.placeholder((N, C, H, W), name='X', dtype=conv_dtype)
        W = te.placeholder((K, C//groups, R, S), name='W', dtype=conv_dtype)
    elif tensor_format == 1:
        X = te.placeholder((N, H, W, C), name='X', dtype=conv_dtype)
        W = te.placeholder((K, R, S, C//groups), name='W', dtype=conv_dtype)
    Y = tvm.contrib.cudnn.conv_forward(X, W, pad, stride, dilation, conv_mode, tensor_format, algo, conv_dtype, groups)
    sched = te.create_schedule(Y.op)
    vendor_kernel = tvm.build(sched, [X, W, Y], CUDATarget)
    return vendor_kernel, Y

class cuDNNConvNCHWFixture:
    def __init__(self, N, C, H, W, K, R, S, conv_dtype='float32', groups=1):
        self.X_np = np.random.uniform(-0.1, 0.1, size=(N, C, H, W)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(K, C//groups, R, S)).astype(np.float32)
        pad = [R//2, S//2]
        stride = [1, 1]
        dilation = [0, 0]
        self.cudnn_kernel, self.Y = \
                _vendor_conv_kernel(N, C, H, W, K, R, S, pad, stride, dilation, 0, 0, -1, conv_dtype, groups=1)

        module_data = self.module_data()
        self.cudnn_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()

    def module_data(self):
        return [tvm.nd.array(self.X_np, device=CUDAContext),
                tvm.nd.array(self.W_np, device=CUDAContext),
                tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                      dtype=np.float32), device=CUDAContext)]

class cuBLASBatchMatmulNTFixture:
    __slots__ = 'B', 'M', 'K', 'N', 'cublas_kernel', \
                'X_np', 'W_np', \
                'Y', 'Y_np_expected'

    def __init__(self, B, M, K, N):
        self.B, self.M, self.K, self.N = B, M, K, N
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, M, K)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(B, N, K)).astype(np.float32)

        self.cublas_kernel, self.Y = \
                _vendor_batch_matmul_kernel(B=B, M=M, K=K, N=N, transpose_b=True)

        module_data = self.module_data()
        self.cublas_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()

    def module_data(self):
        return [tvm.nd.array(self.X_np, device=CUDAContext),
                tvm.nd.array(self.W_np, device=CUDAContext),
                tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                      dtype=np.float32), device=CUDAContext)]


class cuBLASBatchMatmulNNFixture:
    __slots__ = 'B', 'M', 'K', 'N', 'cublas_kernel', \
                'X_np', 'W_np', \
                'Y', 'Y_np_expected'

    def __init__(self, B, M, K, N):
        self.B, self.M, self.K, self.N = B, M, K, N
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, M, K)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(B, K, N)).astype(np.float32)

        self.cublas_kernel, self.Y = \
                _vendor_batch_matmul_kernel(B=B, M=M, K=K, N=N, transpose_b=False)

        module_data = self.module_data()
        self.cublas_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()

    def module_data(self):
        return [tvm.nd.array(self.X_np, device=CUDAContext),
                tvm.nd.array(self.W_np, device=CUDAContext),
                tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                      dtype=np.float32), device=CUDAContext)]
