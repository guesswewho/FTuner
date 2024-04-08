#include <tvm/support/parallel_for.h>
#include <tvm/tir/dyn_shape_var_functor.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "tvm/auto_scheduler/loop_state.h"
#include "tvm/hardware/hw_aligned_config.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>> ThreadsNumberFilter(
    const SearchTask& task, const std::vector<hardware::HwAlignedConfig>& configs,
    const std::vector<State>& cand_states) {
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < configs.size(); i++) {
    if (configs[i].threads_num %
            (task->hardware_api->warp_size * task->hardware_api->compute_sm_partition[1]->value) ==
        0) {
      filtered_states.push_back(cand_states[i]);
      filtered_configs.push_back(configs[i]);
    }
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>> PaddingFilter(
    const SearchTask& task, const std::vector<hardware::HwAlignedConfig>& configs,
    const std::vector<State>& cand_states, const runtime::Array<IntImm> wkl_inst,
    double padding_penalty_threshold) {
  Map<String, IntImm> shape_var_value_map;
  Array<DynShapeVar> shape_vars = task->shape_vars.value();

  CHECK(shape_vars.size() == wkl_inst.size());
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i]->name_hint, wkl_inst[i]);
  }

  // if (enable_verbose_logging) {
  //   LOG(INFO) << MapToString(shape_var_value_map, true);
  // }

  DynShapeVarReplacer replacer([&shape_var_value_map](const DynShapeVarNode* op) -> PrimExpr {
    auto shape_var_value_map_iter = shape_var_value_map.find(op->name_hint);
    if (shape_var_value_map_iter != shape_var_value_map.end()) {
      return (*shape_var_value_map_iter).second;
    }
    LOG(FATAL) << "Dynamic Axis Node " << GetRef<DynShapeVar>(op) << " has not been found in "
               << MapToString(shape_var_value_map);
    return GetRef<DynShapeVar>(op);
  });
  std::vector<bool> valid_states(configs.size());
  support::parallel_for(  // 0
      0, int(configs.size()),
      [&cand_states, &valid_states, &replacer, &padding_penalty_threshold](int index) {
        arith::Analyzer analyzer;
        float padding_penalty = 1.;
        for (const Step& step : cand_states[index]->transform_steps) {
          if (const SplitStepNode* const split_step = step.as<SplitStepNode>()) {
            if (split_step->lengths.size() == 3 || split_step->lengths.size() == 2) {
              int64_t extent = GetIntImm(analyzer.Simplify(replacer(split_step->extent.value())));
              int64_t split_length = 1;

              for (const Optional<Integer>& len : split_step->lengths) {
                split_length *= len.value()->value;
              }

              float padding_ratio = extent * 1. / floor_by(extent, split_length);
              padding_penalty *= padding_ratio;
            }
          }
        }
        if (padding_penalty > padding_penalty_threshold) {
          valid_states[index] = true;
        } else {
          valid_states[index] = false;
        }
      });
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < valid_states.size(); i++) {
    if (valid_states[i]) {
      filtered_states.push_back(cand_states[i]);
      filtered_configs.push_back(configs[i]);
    }
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>> OccupancyFilter(
    const SearchTask& task, const std::vector<hardware::HwAlignedConfig>& configs,
    const std::vector<State>& cand_states, const runtime::Array<IntImm> wkl_inst) {
  Map<String, IntImm> shape_var_value_map;
  Array<DynShapeVar> shape_vars = task->shape_vars.value();

  CHECK(shape_vars.size() == wkl_inst.size());
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i]->name_hint, wkl_inst[i]);
  }

  DynShapeVarReplacer replacer([&shape_var_value_map](const DynShapeVarNode* op) -> PrimExpr {
    auto shape_var_value_map_iter = shape_var_value_map.find(op->name_hint);
    if (shape_var_value_map_iter != shape_var_value_map.end()) {
      return (*shape_var_value_map_iter).second;
    }
    LOG(FATAL) << "Dynamic Axis Node " << GetRef<DynShapeVar>(op) << " has not been found in "
               << MapToString(shape_var_value_map);
    return GetRef<DynShapeVar>(op);
  });
  std::vector<size_t> states_grid_size(configs.size());
  support::parallel_for(  // 0
      0, int(configs.size()), [&cand_states, &states_grid_size, &replacer](int index) {
        arith::Analyzer analyzer;
        size_t grid_size = 1;
        for (const Step& step : cand_states[index]->transform_steps) {
          if (const SplitStepNode* const split_step = step.as<SplitStepNode>()) {
            if (split_step->lengths.size() == 3) {
              int64_t extent = GetIntImm(analyzer.Simplify(replacer(split_step->extent.value())));
              int64_t split_length = 1;

              for (const Optional<Integer>& len : split_step->lengths) {
                split_length *= len.value()->value;
              }
              size_t extent_ratio = floor_div(extent, split_length);
              CHECK(extent_ratio >= 1);
              grid_size *= extent_ratio;
            }
          }
        }
        states_grid_size[index] = grid_size;
      });
  size_t max_grid_size = 0;
  for (int i = 0; i < states_grid_size.size(); i++) {
    if (states_grid_size[i] > max_grid_size) {
      max_grid_size = states_grid_size[i];
    }
  }
  size_t max_sm_times = floor_div(max_grid_size, task->hardware_api->glbmem_sm_partition[0]->value);
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  float occupancy_ratio = 0.95;
  LOG(INFO) << float(task->hardware_api->lt_ratio) << " " << float(task->hardware_api->gt_ratio);
  while (filtered_configs.size() == 0) {
    size_t sm_times = task->hardware_api->smem_sm_partition[1]->value;
    while (sm_times <= max_sm_times) {
      for (int i = 0; i < states_grid_size.size(); i++) {
        float coeff = states_grid_size[i] < static_cast<size_t>(task->hardware_params->num_cores)
                          ? task->hardware_api->lt_ratio
                          : task->hardware_api->gt_ratio;
        float occupancy_penalty = coeff * states_grid_size[i] /
                                  ((coeff - 1) * states_grid_size[i] +
                                   floor_by(states_grid_size[i], task->hardware_params->num_cores));
        if (floor_div(states_grid_size[i], task->hardware_api->glbmem_sm_partition[0]->value) ==
                sm_times &&
            occupancy_penalty > occupancy_ratio) {
          filtered_configs.push_back(configs[i]);
          filtered_states.push_back(cand_states[i]);
        }
        // if (floor_div(states_grid_size[i], task->hardware_api->glbmem_sm_partition[0]->value) ==
        //         sm_times &&
        //     1. * states_grid_size[i] /
        //             floor_by(states_grid_size[i],
        //                      task->hardware_api->glbmem_sm_partition[0]->value) >
        //         occupancy_ratio) {
        //   filtered_configs.push_back(configs[i]);
        //   filtered_states.push_back(cand_states[i]);
        // }
      }
      sm_times++;
    }
    occupancy_ratio -= 0.05;
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>>
RegisterLaunchBoundsFilter(const SearchTask& task,
                           const std::vector<hardware::HwAlignedConfig>& configs,
                           const std::vector<State>& cand_states,
                           const runtime::Array<IntImm> wkl_inst) {
  int rest_reg = 128;
  Map<String, IntImm> shape_var_value_map;
  Array<DynShapeVar> shape_vars = task->shape_vars.value();
  int sch_base = cand_states[0]->stages.size() > 7 ? 2 : 1;
  CHECK(shape_vars.size() == wkl_inst.size());
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i]->name_hint, wkl_inst[i]);
  }

  DynShapeVarReplacer replacer([&shape_var_value_map](const DynShapeVarNode* op) -> PrimExpr {
    auto shape_var_value_map_iter = shape_var_value_map.find(op->name_hint);
    if (shape_var_value_map_iter != shape_var_value_map.end()) {
      return (*shape_var_value_map_iter).second;
    }
    LOG(FATAL) << "Dynamic Axis Node " << GetRef<DynShapeVar>(op) << " has not been found in "
               << MapToString(shape_var_value_map);
    return GetRef<DynShapeVar>(op);
  });
  std::vector<bool> valid_states(configs.size());
  LOG(INFO) << cand_states[0]->stages.size();
  support::parallel_for(  // 0
      0, int(configs.size()),
      [&valid_states, &configs, &cand_states, &replacer, &task, &sch_base](int index) {
        arith::Analyzer analyzer;
        size_t grid_size = 1;
        for (const Step& step : cand_states[index]->transform_steps) {
          if (const SplitStepNode* const split_step = step.as<SplitStepNode>()) {
            if (split_step->lengths.size() == 3) {
              int64_t extent = GetIntImm(analyzer.Simplify(replacer(split_step->extent.value())));
              int64_t split_length = 1;

              for (const Optional<Integer>& len : split_step->lengths) {
                split_length *= len.value()->value;
              }
              size_t extent_ratio = floor_div(extent, split_length);
              CHECK(extent_ratio >= 1);
              grid_size *= extent_ratio;
            }
          }
        }
        size_t blocks_in_sm =
            std::min(size_t(task->hardware_api->smem_sm_partition[1]->value),
                     floor_div(grid_size, task->hardware_api->smem_sm_partition[0]->value));
        if (blocks_in_sm * configs[index].threads_num *
                    (configs[index].single_thread_reg_usage +
                     (configs[index].single_thread_reg_usage * configs[index].reduce_tiles[0][0] *
                      1.0 / 16)) <
                task->hardware_api->max_reg_per_sm &&
            configs[index].single_thread_reg_usage*sch_base +
                    (configs[index].single_thread_reg_usage * configs[index].reduce_tiles[0][0] *
                     1.0 / 16) <
                255) {
          valid_states[index] = true;
        } else {
          valid_states[index] = false;
        }
      });
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < valid_states.size(); i++) {
    if (valid_states[i]) {
      filtered_states.push_back(cand_states[i]);
      filtered_configs.push_back(configs[i]);
    }
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>>
SharedMemoryLaunchBoundsFilter(const SearchTask& task,
                               const std::vector<hardware::HwAlignedConfig>& configs,
                               const std::vector<State>& cand_states,
                               const runtime::Array<IntImm> wkl_inst) {
  Map<String, IntImm> shape_var_value_map;
  Array<DynShapeVar> shape_vars = task->shape_vars.value();

  CHECK(shape_vars.size() == wkl_inst.size());
  for (size_t i = 0; i < shape_vars.size(); ++i) {
    shape_var_value_map.Set(shape_vars[i]->name_hint, wkl_inst[i]);
  }

  DynShapeVarReplacer replacer([&shape_var_value_map](const DynShapeVarNode* op) -> PrimExpr {
    auto shape_var_value_map_iter = shape_var_value_map.find(op->name_hint);
    if (shape_var_value_map_iter != shape_var_value_map.end()) {
      return (*shape_var_value_map_iter).second;
    }
    LOG(FATAL) << "Dynamic Axis Node " << GetRef<DynShapeVar>(op) << " has not been found in "
               << MapToString(shape_var_value_map);
    return GetRef<DynShapeVar>(op);
  });
  std::vector<bool> valid_states(configs.size());
  support::parallel_for(  // 0
      0, int(configs.size()), [&valid_states, &configs, &cand_states, &replacer, &task](int index) {
        arith::Analyzer analyzer;
        size_t grid_size = 1;
        for (const Step& step : cand_states[index]->transform_steps) {
          if (const SplitStepNode* const split_step = step.as<SplitStepNode>()) {
            if (split_step->lengths.size() == 3) {
              int64_t extent = GetIntImm(analyzer.Simplify(replacer(split_step->extent.value())));
              int64_t split_length = 1;

              for (const Optional<Integer>& len : split_step->lengths) {
                split_length *= len.value()->value;
              }
              size_t extent_ratio = floor_div(extent, split_length);
              CHECK(extent_ratio >= 1);
              grid_size *= extent_ratio;
            }
          }
        }
        size_t blocks_in_sm =
            std::min(size_t(task->hardware_api->smem_sm_partition[1]->value),
                     floor_div(grid_size, task->hardware_api->smem_sm_partition[0]->value));
        if (blocks_in_sm * configs[index].smem_usage < task->hardware_api->max_smem_usage_per_sm) {
          valid_states[index] = true;
        } else {
          valid_states[index] = false;
        }
        if (blocks_in_sm * configs[index].smem_usage > task->hardware_api->max_smem_usage_per_sm) {
          valid_states[index] = false;
        }
      });
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < valid_states.size(); i++) {
    if (valid_states[i]) {
      filtered_states.push_back(cand_states[i]);
      filtered_configs.push_back(configs[i]);
    }
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>>
SharedMemoryComputeIntensiveFilter(const SearchTask& task,
                                   const std::vector<hardware::HwAlignedConfig>& configs,
                                   const std::vector<State>& cand_states) {
  // double compute_intensive_threshold = 1;
  // std::vector<hardware::HwAlignedConfig> filtered_configs;
  // std::vector<State> filtered_states;
  // while (filtered_configs.size() < 20 || filtered_configs.size() < configs.size()) {
  //   for (int i = 0; i < configs.size(); i++) {
  //     if (configs[i].compute_intensive_ratio[0] > compute_intensive_threshold) {
  //       filtered_configs.push_back(configs[i]);
  //       filtered_states.push_back(cand_states[i]);
  //     }
  //   }
  //   compute_intensive_threshold -= 0.1;
  // }
  std::vector<std::pair<hardware::HwAlignedConfig, int>> configs_for_sort;
  for (int i = 0; i < configs.size(); i++) {
    configs_for_sort.push_back(std::make_pair(configs[i], i));
  }
  static auto cmp_compute_ratio = [](const std::pair<hardware::HwAlignedConfig, int>& a,
                                     const std::pair<hardware::HwAlignedConfig, int>& b) {
    return a.first.compute_intensive_ratio[0] > b.first.compute_intensive_ratio[0];
  };
  std::sort(configs_for_sort.begin(), configs_for_sort.end(), cmp_compute_ratio);
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < std::min(20, int(configs_for_sort.size())); i++) {
    filtered_configs.push_back(configs_for_sort[i].first);
    filtered_states.push_back(cand_states[configs_for_sort[i].second]);
  }
  LOG(INFO) << filtered_configs.size();
  return std::make_pair(filtered_configs, filtered_states);
  // std::vector<std::pair<hardware::HwAlignedConfig, int>> configs_for_sort;
  // for (int i = 0; i < configs.size(); i++) {
  //   configs_for_sort.push_back(std::make_pair(configs[i], i));
  // }
  // static auto cmp_compute_ratio = [](const std::pair<hardware::HwAlignedConfig, int>& a,
  //                                    const std::pair<hardware::HwAlignedConfig, int>& b) {
  //   return a.first.compute_intensive_ratio[0] > b.first.compute_intensive_ratio[0];
  // };
  // std::sort(configs_for_sort.begin(), configs_for_sort.end(), cmp_compute_ratio);
  // std::vector<hardware::HwAlignedConfig> filtered_configs;
  // std::vector<State> filtered_states;
  // for (int i = 0; i < std::min(1000, int(configs_for_sort.size())); i++) {
  //   filtered_configs.push_back(configs_for_sort[i].first);
  //   filtered_states.push_back(cand_states[configs_for_sort[i].second]);
  // }
  // LOG(INFO) << filtered_configs.size();
  // return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>>
RegComputeIntensiveFilter(const SearchTask& task,
                          const std::vector<hardware::HwAlignedConfig>& configs,
                          const std::vector<State>& cand_states,
                          const runtime::Array<IntImm> wkl_inst) {
  // double compute_intensive_threshold = 1;
  // std::vector<hardware::HwAlignedConfig> filtered_configs;
  // std::vector<State> filtered_states;
  // while (filtered_configs.size() == 0) {
  //   for (int i = 0; i < configs.size(); i++) {
  //     if (configs[i].compute_intensive_ratio[1] > compute_intensive_threshold) {
  //       filtered_configs.push_back(configs[i]);
  //       filtered_states.push_back(cand_states[i]);
  //     }
  //   }
  //   compute_intensive_threshold -= 0.1;
  // }
  std::vector<std::pair<hardware::HwAlignedConfig, int>> configs_for_sort;
  for (int i = 0; i < configs.size(); i++) {
    configs_for_sort.push_back(std::make_pair(configs[i], i));
  }
  static auto cmp_compute_ratio = [](const std::pair<hardware::HwAlignedConfig, int>& a,
                                     const std::pair<hardware::HwAlignedConfig, int>& b) {
    return a.first.compute_intensive_ratio[1] > b.first.compute_intensive_ratio[1];
  };
  std::sort(configs_for_sort.begin(), configs_for_sort.end(), cmp_compute_ratio);
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < std::min(10, int(configs_for_sort.size())); i++) {
    filtered_configs.push_back(configs_for_sort[i].first);
    filtered_states.push_back(cand_states[configs_for_sort[i].second]);
  }
  LOG(INFO) << filtered_configs.size();
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>>
SpaceProductionThresholdFilter(const SearchTask& task,
                               const std::vector<hardware::HwAlignedConfig>& configs,
                               const std::vector<State>& cand_states,
                               const runtime::Array<IntImm> wkl_inst) {
  std::vector<std::pair<hardware::HwAlignedConfig, int>> configs_for_sort;
  for (int i = 0; i < configs.size(); i++) {
    configs_for_sort.push_back(std::make_pair(configs[i], i));
  }
  static auto cmp_ij = [](const std::pair<hardware::HwAlignedConfig, int>& a,
                          const std::pair<hardware::HwAlignedConfig, int>& b) {
    return a.first.space_production_threshold > b.first.space_production_threshold;
  };
  std::sort(configs_for_sort.begin(), configs_for_sort.end(), cmp_ij);
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < std::min(10, int(configs_for_sort.size())); i++) {
    filtered_configs.push_back(configs_for_sort[i].first);
    filtered_states.push_back(cand_states[configs_for_sort[i].second]);
  }
  return std::make_pair(filtered_configs, filtered_states);
}

inline std::pair<std::vector<hardware::HwAlignedConfig>, std::vector<State>> KThresholdFilter(
    const SearchTask& task, const std::vector<hardware::HwAlignedConfig>& configs,
    const std::vector<State>& cand_states, const runtime::Array<IntImm> wkl_inst) {
  std::vector<std::pair<hardware::HwAlignedConfig, int>> configs_for_sort;
  for (int i = 0; i < configs.size(); i++) {
    configs_for_sort.push_back(std::make_pair(configs[i], i));
  }
  static auto cmp_k = [](const std::pair<hardware::HwAlignedConfig, int>& a,
                         const std::pair<hardware::HwAlignedConfig, int>& b) {
    return std::accumulate(a.first.k_threshold.begin(), a.first.k_threshold.end(), 1,
                           std::multiplies<double>()) <
           std::accumulate(b.first.k_threshold.begin(), b.first.k_threshold.end(), 1,
                           std::multiplies<double>());
  };
  std::sort(configs_for_sort.begin(), configs_for_sort.end(), cmp_k);
  std::vector<hardware::HwAlignedConfig> filtered_configs;
  std::vector<State> filtered_states;
  for (int i = 0; i < std::min(10, int(configs_for_sort.size())); i++) {
    filtered_configs.push_back(configs_for_sort[i].first);
    filtered_states.push_back(cand_states[configs_for_sort[i].second]);
  }
  return std::make_pair(filtered_configs, filtered_states);
}

}  // namespace auto_scheduler
}  // namespace tvm