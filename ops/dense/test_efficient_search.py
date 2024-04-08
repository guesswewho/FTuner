from tvm import tir

import logging
import numpy

logger = logging.getLogger(__name__)

from ...shared import tvm_dev_decor
from ..shared.auto_scheduler import DietCodeAutoScheduler
from ..shared.utils import cross_product
from .fixture import Dense, cuBLASDenseFixture, Gemm, DenseAdd
from tvm.hardware import K80, HardwareAPI, V100, RTX3090

auto_scheduler = DietCodeAutoScheduler()


@tvm_dev_decor
def test_train(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=[(T, I, H)],
                         wkl_inst_weights=[1.],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}x{}x{}'.format(B * T, I, H)
                         )


@tvm_dev_decor
def test_infer(pytestconfig):
    B = pytestconfig.getoption('B')
    T = pytestconfig.getoption('T')
    I = pytestconfig.getoption('I')
    H = pytestconfig.getoption('H')

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.infer(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=[(T, I, H)],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_log_fname=pytestconfig.getoption('sched_log_fname')
                         )


@tvm_dev_decor
def test_train_dynT():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_group_dynT(pytestconfig):
    group = pytestconfig.getoption('group')
    print(group)
    print("---------------------")
    B = 16
    T = list(range(1, group))
    T.append(group)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_5_dynT():
    B = 16
    T = []
    T.append(5)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_24_dynT():
    B = 16
    T = []
    T.append(24)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_43_dynT():
    B = 16
    T = []
    T.append(43)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_62_dynT():
    B = 16
    T = []
    T.append(62)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_81_dynT():
    B = 16
    T = []
    T.append(81)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_100_dynT():
    B = 16
    T = []
    T.append(100)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_119_dynT():
    B = 16
    T = []
    T.append(119)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_128_dynT():
    B = 16
    T = []
    T.append(128)
    I = 768
    H = 2304
    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B*DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_fine_grained_dynT():
    B = 16
    T = list(range(1, 128))
    T.append(128)
    I = 768
    H = 2304

    wkl_insts = cross_product(T, (I, H))
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx{}x{}'.format(B, I, H),
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

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')

    auto_scheduler.infer(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_log_fname="/mnt/ops/dense/dietcode_autosched_dense_16xTx768x2304.json"
                         )


@tvm_dev_decor
def test_train_BERT_H768():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 768), (3072, 768)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTxIx768'.format(B),
                         hardware_api=hardware_api
                         )


@tvm_dev_decor
def test_train_BERT_H3072():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 3072)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx768x3072'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_BERT_LARGE_I1024_H4096():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(1024, 4096)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx1024x4096'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_BERT_LARGE_I1024_H1024():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(1024, 1024)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx1024x1024'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_BERT_LARGE_I4096_H1024():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(4096, 1024)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                     wkl_func_args=(B * DynT, DynI, DynH),
                     shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                     wkl_inst_weights=[1. for _ in wkl_insts],
                     fcublas_fixture=cuBLASDenseFixture,
                     sched_func_name_prefix='dense_{}xTx4096x1024'.format(B),
                     hardware_api=hardware_api
                     )

@tvm_dev_decor
def test_train_GPT_I768_H2304():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 2304)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=DenseAdd,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_add_{}xTx768x2304'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_GPT_I3072_H768():
    B = 16
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(3072, 768)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=DenseAdd,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_add_{}xTx3072x768'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_GPT_I768_H2304():
    B = 16
    T = []
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 2304)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=DenseAdd,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_add_{}xTx768x2304'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_GPT_I768_H3072():
    B = 16
    T = []
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 3072)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=DenseAdd,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_add_{}xTx768x3072'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_GPT_I768_H768():
    B = 16
    T = []
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 768)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=DenseAdd,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_add_{}xTx768x768'.format(B),
                         hardware_api=hardware_api
                         )

@tvm_dev_decor
def test_train_GPT_I768_H50257():
    B = 16
    T = []
    T = list(range(5, 128, 19))
    T.append(128)

    wkl_insts = cross_product(T, [(768, 50257)])
    logger.info("Workload Instances: {}".format(wkl_insts))

    DynT, DynI, DynH = tir.DynShapeVar('T'), tir.DynShapeVar('I'), tir.DynShapeVar('H')
    hardware_api = HardwareAPI(V100())
    auto_scheduler.train(wkl_func=Dense,
                         wkl_func_args=(B * DynT, DynI, DynH),
                         shape_vars=[DynT, DynI, DynH], wkl_insts=wkl_insts,
                         wkl_inst_weights=[1. for _ in wkl_insts],
                         fcublas_fixture=cuBLASDenseFixture,
                         sched_func_name_prefix='dense_{}xTx768x50257'.format(B),
                         hardware_api=hardware_api
                         )