#ifndef TVM_HARDWARE_HARDWARE_API_H_
#define TVM_HARDWARE_HARDWARE_API_H_

#include <tvm/runtime/ndarray.h>

#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"

namespace tvm {
namespace hardware {

class HardwareAPINode : public Object {
 public:
  int num_level;
  Array<IntImm> bandwidth;
  double peak_flops;
  Array<IntImm> limit;
  Array<IntImm> reg_cap;
  Array<IntImm> smem_cap;
  Array<IntImm> compute_max_core;
  Array<IntImm> mem_max_core;
  bool para_opt;
  int warp_size;
  Array<IntImm> compute_sm_partition;
  Array<IntImm> smem_sm_partition;
  Array<String> compute_block_schedule_way;
  Array<String> smem_block_schedule_way;
  Array<IntImm> transaction_size;
  Array<IntImm> glbmem_sm_partition;
  int smem_bank_size;
  int bank_number;
  String compute_capability;
  int max_smem_usage_per_sm;
  int max_reg_per_sm;
  float lt_ratio;
  float gt_ratio;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_level", &num_level);
    v->Visit("bandwidth", &bandwidth);
    v->Visit("peak_flops", &peak_flops);
    v->Visit("limit", &limit);
    v->Visit("reg_cap", &reg_cap);
    v->Visit("smem_cap", &smem_cap);
    v->Visit("compute_max_core", &compute_max_core);
    v->Visit("mem_max_core", &mem_max_core);
    v->Visit("para_opt", &para_opt);
    v->Visit("warp_size", &warp_size);
    v->Visit("compute_sm_partition", &compute_sm_partition);
    v->Visit("smem_sm_partition", &smem_sm_partition);
    v->Visit("compute_block_schedule_way", &compute_block_schedule_way);
    v->Visit("smem_block_schedule_way", &smem_block_schedule_way);
    v->Visit("transaction_size", &transaction_size);
    v->Visit("glbmem_sm_partition", &glbmem_sm_partition);
    v->Visit("smem_bank_size", &smem_bank_size);
    v->Visit("bank_number", &bank_number);
    v->Visit("compute_capability", &compute_capability);
    v->Visit("max_smem_usage_per_sm", &max_smem_usage_per_sm);
    v->Visit("max_reg_per_sm", &max_reg_per_sm);
  }

  IntImm MemoryBw(int mem_level);
  double PeakFlops();
  IntImm RegCap(int mem_level);
  IntImm MemCap(int mem_level);

  TVM_DECLARE_FINAL_OBJECT_INFO(HardwareAPINode, Object);
};

class HardwareAPI : public ObjectRef {
 public:
  HardwareAPI(int num_level, Array<IntImm> bandwidth = Array<IntImm>(), double peak_flops = 0,
              Array<IntImm> limit = Array<IntImm>(), Array<IntImm> reg_cap = Array<IntImm>(),
              Array<IntImm> smem_cap = Array<IntImm>(),
              Array<IntImm> compute_max_core = Array<IntImm>(),
              Array<IntImm> mem_max_core = Array<IntImm>(), bool para_opt = false,
              int warp_size = 0, Array<IntImm> compute_sm_partition = Array<IntImm>(),
              Array<IntImm> smem_sm_partition = Array<IntImm>(),
              Array<String> compute_block_schedule_way = Array<String>(),
              Array<String> smem_block_schedule_way = Array<String>(),
              Array<IntImm> transaction_size = Array<IntImm>(),
              Array<IntImm> glbmem_sm_partition = Array<IntImm>(), int smem_bank_size = 0,
              int bank_number = 0, String compute_capability = "", int max_smem_usage_per_sm=0,
              int max_reg_per_sm=0, double lt_ratio=1.0, double gt_ratio=1.0);
  TVM_DEFINE_OBJECT_REF_METHODS(HardwareAPI, ObjectRef, HardwareAPINode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(HardwareAPINode);
};
}  // namespace hardware

}  // namespace tvm

#endif