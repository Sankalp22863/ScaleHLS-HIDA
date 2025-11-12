#include "scalehls/Transforms/CustomPipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace codesignhls {
namespace {

struct LowerMaximumFPass
    : public PassWrapper<LowerMaximumFPass, OperationPass<func::FuncOp>> {
  // MLIR 16: explicit TypeID is required if the class is in an anonymous namespace
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMaximumFPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    OpBuilder b(f);
    int64_t rewrites = 0;

    f.walk([&](Operation *op) {
      // We can’t include arith::MaximumFOp on MLIR16; match by name.
      if (op->getName().getStringRef() != "arith.maximumf")
        return;

      Value lhs = op->getOperand(0), rhs = op->getOperand(1);
      auto loc = op->getLoc();

      // MLIR16 API: result type of CmpFOp is inferred (i1 or tensor<i1>)
      Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
      Value sel = b.create<arith::SelectOp>(loc, cmp, lhs, rhs);

      op->getResult(0).replaceAllUsesWith(sel);
      op->erase();
      ++rewrites;
    });

    llvm::errs() << "[LowerMaximumF] " << f.getName() << ": rewrote "
                 << rewrites << " arith.maximumf ops\n";
  }
};

} // End anonymous namespace

std::unique_ptr<mlir::Pass> createLowerMaximumFPass() {
  return std::make_unique<LowerMaximumFPass>();
}

} // End codesignhls namespace
