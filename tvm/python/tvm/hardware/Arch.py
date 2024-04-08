class Arch:
    # compute 3.7
    def __init__(self):
        super().__init__()
        self.num_level = 0
        self.bandwidth = [] # measured
        # compute throughput in GFLOPS
        self.peak_flops = 0 # found in hardware spec TODO
        self.limit = []
        self.reg_cap = [] # max reg per block(from hw: 65536), max reg per thread TODO 0 get device, 1 what?
        self.smem_cap = [] # device query, max shared memory per block TODO get device?
        self.compute_max_core = []
        self.mem_max_core = []
        self.para_opt = False

        self.warp_size = 0
        self.compute_sm_partition = [] # TODO verify?
        self.smem_sm_partition = []
        self.compute_block_schedule_way = []
        self.smem_block_schedule_way = []
        self.transaction_size = []   # TODO: global memory
        self.glbmem_sm_partition = []   # TODO 32: The number of warps per sm when global memory reaches the peak throughput
        self.smem_bank_size = 0
        self.bank_number = 0
        self.compute_capability = ''
        self.max_smem_usage = 0
        self.max_reg_per_sm = 0
        self.lt_ratio = 1
        self.gt_ratio = 1