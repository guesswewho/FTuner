from .K80 import *
from .Arch import *
from .V100 import *
from .RTX3090 import *
from . import _ffi_api
from tvm.runtime import Object

class HardwareAPI(Object):
    def __init__(self, arch) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.HardwareAPI,
            arch.num_level,
            arch.bandwidth,
            arch.peak_flops,
            arch.limit,
            arch.reg_cap,
            arch.smem_cap,
            arch.compute_max_core,
            arch.mem_max_core,
            arch.para_opt,
            arch.warp_size,
            arch.compute_sm_partition,
            arch.smem_sm_partition,
            arch.compute_block_schedule_way,
            arch.smem_block_schedule_way,
            arch.transaction_size,
            arch.glbmem_sm_partition,
            arch.smem_bank_size,
            arch.bank_number,
            arch.compute_capability,
            arch.max_smem_usage,
            arch.max_reg_per_sm,
            arch.lt_ratio,
            arch.gt_ratio
        )