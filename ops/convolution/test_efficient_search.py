from tvm import tir

import logging

logger = logging.getLogger(__name__)

from ...shared import tvm_dev_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from ..shared.utils import cross_product
from .fixture import ConvNCHW, cuDNNConvNCHWFixture
from tvm.hardware import K80, HardwareAPI, V100, RTX3090

import random

auto_scheduler = DietCodeAutoScheduler()


@tvm_dev_decor
def test_train_dynT():
    N = 16
    C = 128
    H = list(range(320, 640, 19))+[640]
    W = list(range(320, 640, 19))+[640]
    K = [64, 128, 256, 512]
    R = [1, 3, 5, 7]
    S = [1, 3, 5, 7]
    img_shape = cross_product(H, W)
    kernel_shape = list(zip(R, S))
    input_data = cross_product((N, C), img_shape)
    weight = cross_product(K, kernel_shape)
    wkl_insts = cross_product(input_data, weight)
    wkl_insts = random.sample(wkl_insts, 8)
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynN, DynC, DynH, DynW, DynK, DynR, DynS = tir.DynShapeVar('N'), tir.DynShapeVar('C'), tir.DynShapeVar('H'), tir.DynShapeVar('W'), tir.DynShapeVar('K'), tir.DynShapeVar('R'), tir.DynShapeVar('S')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=ConvNCHW,
                         wkl_func_args=(DynN, DynC, DynH, DynW, DynK, DynR, DynS),
                         shape_vars=[DynN, DynC, DynH, DynW, DynK, DynR, DynS], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuDNNConvNCHWFixture,
                         sched_func_name_prefix='Conv2d_NxCxHxWxKxRxS',
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_infer_dynT(pytestconfig):
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)
    I = 768
    H = 2304

    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynN, DynC, DynH, DynW, DynK, DynR, DynS = tir.DynShapeVar('N'), tir.DynShapeVar('C'), tir.DynShapeVar('H'), tir.DynShapeVar('W'), tir.DynShapeVar('K'), tir.DynShapeVar('R'), tir.DynShapeVar('S')

    auto_scheduler.infer(wkl_func=ConvNCHW,
                         wkl_func_args=(DynN, DynC, DynH, DynW, DynK, DynR, DynS),
                         shape_vars=[DynN, DynC, DynH, DynW, DynK, DynR, DynS], wkl_insts=wkl_insts,
                         fcublas_fixture=cuDNNConvNCHWFixture,
                         sched_log_fname="/mnt/ops/convolution/dietcode_autosched_Conv2d_NxCxHxWxKxRxS.json"
                         )