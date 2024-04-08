#include <tvm/hardware/hardware_api.h>
#include <tvm/runtime/registry.h>

#include <utility>

#include "tvm/ir/expr.h"
#include "tvm/runtime/memory.h"

namespace tvm {
namespace hardware {

TVM_REGISTER_NODE_TYPE(HardwareAPINode);

HardwareAPI::HardwareAPI(int num_level, Array<IntImm> bandwidth, double peak_flops,
                         Array<IntImm> limit, Array<IntImm> reg_cap, Array<IntImm> smem_cap,
                         Array<IntImm> compute_max_core, Array<IntImm> mem_max_core, bool para_opt,
                         int warp_size, Array<IntImm> compute_sm_partition,
                         Array<IntImm> smem_sm_partition, Array<String> compute_block_schedule_way,
                         Array<String> smem_block_schedule_way, Array<IntImm> transaction_size,
                         Array<IntImm> glbmem_sm_partition, int smem_bank_size, int bank_number,
                         String compute_capability, int max_smem_usage_per_sm, int max_reg_per_sm,
                         double lt_ratio, double gt_ratio) {
  auto node = make_object<HardwareAPINode>();
  node->num_level = std::move(num_level);
  node->bandwidth = std::move(bandwidth);
  node->peak_flops = std::move(peak_flops);
  node->limit = std::move(limit);
  node->reg_cap = std::move(reg_cap);
  node->smem_cap = std::move(smem_cap);
  node->compute_max_core = std::move(compute_max_core);
  node->mem_max_core = std::move(mem_max_core);
  node->para_opt = std::move(para_opt);
  node->warp_size = std::move(warp_size);
  node->compute_sm_partition = std::move(compute_sm_partition);
  node->smem_sm_partition = std::move(smem_sm_partition);
  node->compute_block_schedule_way = std::move(compute_block_schedule_way);
  node->smem_block_schedule_way = std::move(smem_block_schedule_way);
  node->transaction_size = std::move(transaction_size);
  node->glbmem_sm_partition = std::move(glbmem_sm_partition);
  node->smem_bank_size = std::move(smem_bank_size);
  node->bank_number = std::move(bank_number);
  node->compute_capability = std::move(compute_capability);
  node->max_smem_usage_per_sm = std::move(max_smem_usage_per_sm);
  node->max_reg_per_sm = std::move(max_reg_per_sm);
  node->lt_ratio = lt_ratio;
  node->gt_ratio = gt_ratio;
  data_ = std::move(node);
}

IntImm HardwareAPINode::MemoryBw(int mem_level) { return this->bandwidth[mem_level]; }

double HardwareAPINode::PeakFlops() { return this->peak_flops; }

IntImm HardwareAPINode::RegCap(int mem_level) { return this->reg_cap[mem_level]; }

IntImm HardwareAPINode::MemCap(int mem_level) { return this->smem_cap[mem_level]; }

TVM_REGISTER_GLOBAL("hardware.HardwareAPI")
    .set_body_typed([](int num_level, Array<IntImm> bandwidth, double peak_flops,
                       Array<IntImm> limit, Array<IntImm> reg_cap, Array<IntImm> smem_cap,
                       Array<IntImm> compute_max_core, Array<IntImm> mem_max_core, bool para_opt,
                       int warp_size, Array<IntImm> compute_sm_partition,
                       Array<IntImm> smem_sm_partition, Array<String> compute_block_schedule_way,
                       Array<String> smem_block_schedule_way, Array<IntImm> transaction_size,
                       Array<IntImm> glbmem_sm_partition, int smem_bank_size, int bank_number,
                       String compute_capability, int max_smem_usage_per_sm, int max_reg_per_sm,
                       double lt_ratio, double gt_ratio) {
      return HardwareAPI(num_level, bandwidth, peak_flops, limit, reg_cap, smem_cap,
                         compute_max_core, mem_max_core, para_opt, warp_size, compute_sm_partition,
                         smem_sm_partition, compute_block_schedule_way, smem_block_schedule_way,
                         transaction_size, glbmem_sm_partition, smem_bank_size, bank_number,
                         compute_capability, max_smem_usage_per_sm, max_reg_per_sm, lt_ratio,
                         gt_ratio);
    });

}  // namespace hardware
}  // namespace tvm