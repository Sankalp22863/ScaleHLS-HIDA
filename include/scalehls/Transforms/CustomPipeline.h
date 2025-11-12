#pragma once
#include "mlir/Pass/Pass.h"

namespace codesignhls {
std::unique_ptr<mlir::Pass> createLowerMaximumFPass();
} 