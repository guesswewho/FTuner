class K80:
    # compute 3.7
    def __init__(self, para_opt = True):
        super().__init__()
        self.num_level = 2
        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        # bandwidth in GBps
        self.bandwidth = [162, 1962] # measured
        # compute throughput in GFLOPS
        self.peak_flops = 1952 # found in hardware spec TODO
        self.limit = []
        self.reg_cap = [32768, 128] # max reg per block(from hw: 65536), max reg per thread TODO 0 get device, 1 what?
        self.smem_cap = [49152] # device query, max shared memory per block TODO get device?
        self.compute_max_core = [13, 13 * 4 * 32]
        self.mem_max_core = [13, 13 * 4 * 32]
        self.para_opt = para_opt

        self.warp_size = 32
        self.compute_sm_partition = [13, 4] # TODO verify?
        self.smem_sm_partition = [13, 2]
        self.compute_block_schedule_way = ["warp", "active block"]
        self.smem_block_schedule_way = ["warp", "active block"]
        self.transaction_size = [32, 256]   # TODO: global memory
        self.glbmem_sm_partition = [13, 32]   # TODO 32: The number of warps per sm when global memory reaches the peak throughput
        self.smem_bank_size = 8
        self.bank_number = 32
        self.compute_capability = 'compute_37'

        # for active block estimation
        self.max_active_blocks = 32
        self.max_smem_usage = 112*1024
        self.max_threads_per_sm = 1536
        self.max_reg_per_sm = 65536
        self.lt_ratio = 1
        self.gt_ratio = 1