#ifndef TVM_HW_ALIGNED_CONFIG_H_
#define TVM_HW_ALIGNED_CONFIG_H_

#include <tvm/runtime/ndarray.h>
#include <tvm/tir/expr_functor.h>

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace hardware {

class HwAlignedConfig {
 public:
  std::vector<std::vector<int>> space_tiles;
  std::vector<std::vector<int>> reduce_tiles;
  std::vector<double> k_threshold;
  std::vector<double> compute_intensive_ratio;
  int single_thread_reg_usage;
  int space_production_threshold;
  int smem_usage;
  int threads_num;

  bool operator<(const HwAlignedConfig& config) const{
    for(int i=0;i<this->space_tiles.size();i++){
      for(int j=0;j<this->space_tiles[i].size();j++){
        if(this->space_tiles[i][j]<config.space_tiles[i][j]){
          return true;
        }else if(this->space_tiles[i][j]>config.space_tiles[i][j]){
          return false;
        }
      }
    }
    for(int i=0;i<this->reduce_tiles.size();i++){
      for(int j=0;j<this->reduce_tiles[i].size();j++){
        if(this->reduce_tiles[i][j]<config.reduce_tiles[i][j]){
          return true;
        }else if(this->reduce_tiles[i][j]>config.reduce_tiles[i][j]){
          return false;
        }
      }
    }
    return false;
  }
};
}  // namespace hardware

}  // namespace tvm

#endif