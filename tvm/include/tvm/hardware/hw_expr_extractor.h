#ifndef TVM_HW_EXPR_EXTRACTOR_H_
#define TVM_HW_EXPR_EXTRACTOR_H_

#include <tvm/runtime/ndarray.h>
#include <tvm/tir/expr_functor.h>

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"

namespace tvm {
using namespace tvm::tir;
namespace hardware {

class HwExprExtractor : public ExprFunctor<bool(const PrimExpr&)> {
 public:
  using ExprFunctor::VisitExpr;
  Array<DataProducer> expr_producer;
  Array<Array<PrimExpr>> expr_indices;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("expr_indices", &expr_indices);
    v->Visit("expr_producer", &expr_producer);
  }

 protected:
  using ExprFunctor::VisitExpr_;
  bool VisitExpr_(const ReduceNode* op) {
    for (auto expr : op->source) {
      if (!VisitExpr(expr)) {
        return false;
      }
    }
    return true;
  }
  template <typename T>
  bool VisitBinary(const T* op) {
    return VisitExpr(op->a) && VisitExpr(op->b);
  }
  bool VisitExpr_(const MulNode* op) { return VisitBinary(op); }
  bool VisitExpr_(const ProducerLoadNode* op) {
    expr_producer.push_back(op->producer);
    expr_indices.push_back(op->indices);
    return true;
  }
};
}  // namespace hardware

}  // namespace tvm

#endif