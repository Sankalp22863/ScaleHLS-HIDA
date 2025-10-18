//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Support/FileUtilities.h"
#include "scalehls/Transforms/Explorer.h"
#include "scalehls/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include <numeric>
#include <random>
#include <iostream>
// #include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include <atomic>
#include <filesystem>
#include <iostream>
// #include "GPModel.h"
// #include <pthread.h>

#define DEBUG_TYPE "scalehls"

using namespace mlir;
using namespace scalehls;

/// Update paretoPoints to remove design points that are not pareto frontiers.
template <typename DesignPointType>
static void updateParetoPoints(SmallVector<DesignPointType, 16> &paretoPoints) {
  // Sort the pareto points with in an ascending order of latency and the an
  // ascending order of dsp number.
  auto latencyThenDspNum = [&](const DesignPointType &a,
                               const DesignPointType &b) {
    return (a.latency < b.latency ||
            (a.latency == b.latency && a.dspNum < b.dspNum));
  };
  llvm::sort(paretoPoints, latencyThenDspNum);

  // Find pareto frontiers. After the sorting, the first design point must be a
  // pareto point.
  auto paretoPoint = paretoPoints[0];
  auto paretoLatency = paretoPoint.latency;
  auto paretoDspNum = paretoPoint.dspNum;
  SmallVector<DesignPointType, 16> frontiers;

  for (auto point : paretoPoints) {
    auto tmpLatency = point.latency;
    auto tmpDspNum = point.dspNum;

    if (tmpDspNum < paretoDspNum) {
      frontiers.push_back(point);

      paretoPoint = point;
      paretoLatency = tmpLatency;
      paretoDspNum = tmpDspNum;

    } else if (tmpDspNum == paretoDspNum && tmpLatency == paretoLatency)
      frontiers.push_back(point);
  }

  paretoPoints = frontiers;
}

//===----------------------------------------------------------------------===//
// LoopDesignSpace Class Definition
//===----------------------------------------------------------------------===//

static void emitTileListDebugInfo(FactorList tileList) {
  LLVM_DEBUG(llvm::dbgs() << "(";
             for (unsigned i = 0, e = tileList.size(); i < e; ++i) {
               llvm::dbgs() << tileList[i];
               if (i != e - 1)
                 llvm::dbgs() << ",";
               else
                 llvm::dbgs() << ") ";
             });
}

LoopDesignSpace::LoopDesignSpace(func::FuncOp func, AffineLoopBand &band,
                                 ScaleHLSEstimator &estimator,
                                 unsigned maxDspNum, unsigned maxExplParallel,
                                 unsigned maxLoopParallel, bool directiveOnly)
    : func(func), band(band), estimator(estimator), maxDspNum(maxDspNum) {
  // Initialize tile vector related members.
  validTileConfigNum = 1;
  for (auto loop : band) {
    auto optionalTripCount = getConstantTripCount(loop);
    if (!optionalTripCount)
      loop.emitError("has variable loop bound");

    unsigned tripCount = optionalTripCount.value();
    tripCountList.push_back(tripCount);

    SmallVector<unsigned, 8> validSizes;
    unsigned size = 1;
    while (size <= std::min(tripCount, maxLoopParallel)) {
      // Push back the current size.
      validSizes.push_back(size);

      // Find the next possible size.
      ++size;
      while (size <= tripCount && tripCount % size != 0)
        ++size;
    }

    validTileSizesList.push_back(validSizes);
    validTileConfigNum *= validSizes.size();
  }

  // The last design point (all loops are fully unrolled) is removed.
  --validTileConfigNum;

  for (TileConfig config = 0; config < validTileConfigNum; ++config) {
    auto tileList = getTileList(config);

    // If the overall parallelism is out of bound, continue to next config.
    auto parallel = std::accumulate(tileList.begin(), tileList.end(),
                                    (unsigned)1, std::multiplies<unsigned>());
    if (parallel > maxExplParallel)
      continue;

    // In only directive opt should be applied, once one loop is unrolled, all
    // innter loops should be fully unrolled.
    if (directiveOnly) {
      bool mustFullyUnroll = false;
      bool isInvalid = false;
      unsigned i = 0;

      for (auto tile : tileList) {
        if (mustFullyUnroll && tile != tripCountList[i])
          isInvalid = true;
        if (tile != 1)
          mustFullyUnroll = true;
        ++i;
      }

      if (isInvalid)
        continue;
    }

    unestimatedTileConfigs.insert(config);
  }
}

/// Return the actual tile vector given a tile config.
FactorList LoopDesignSpace::getTileList(TileConfig config) {
  assert(config < validTileConfigNum && "invalid tile config");

  FactorList tileList;
  unsigned factor = 1;
  for (auto validSizes : validTileSizesList) {
    auto idx = config / factor % validSizes.size();
    factor *= validSizes.size();

    auto size = validSizes[idx];
    tileList.push_back(size);
  }
  return tileList;
}

/// Return the corresponding tile config given a tile list.
TileConfig LoopDesignSpace::getTileConfig(FactorList tileList) {
  assert(tileList.size() == validTileSizesList.size() && "invalid tile list");

  TileConfig config = 0;
  unsigned factor = 1;
  for (unsigned i = 0, e = tileList.size(); i < e; ++i) {
    auto tile = tileList[i];
    auto validSizes = validTileSizesList[i];

    auto idx = llvm::find(validSizes, tile) - validSizes.begin();

    assert(idx >= 0 && idx < (long)validSizes.size() && "invalid tile list");

    config += factor * idx;
    factor *= validSizes.size();
  }

  return config;
}

/// Calculate the Euclid distance of config a and config b.
float LoopDesignSpace::getTileConfigDistance(TileConfig configA,
                                             TileConfig configB) {
  assert(configA < validTileConfigNum && configB < validTileConfigNum &&
         "invalid tile config");

  int64_t distanceSquare = 0;
  unsigned factor = 1;
  for (auto validSizes : validTileSizesList) {
    int64_t idxA = configA / factor % validSizes.size();
    int64_t idxB = configB / factor % validSizes.size();
    factor *= validSizes.size();

    auto idxDistance = idxA - idxB;
    distanceSquare += idxDistance * idxDistance;
  }

  return sqrtf(distanceSquare);
}

/// Evaluate all design points under the given tile config.
bool LoopDesignSpace::evaluateTileConfig(TileConfig config) {
  // If the current tile config is already estimated, return false.
  if (!unestimatedTileConfigs.count(config))
    return false;

  // Annotate the current tile config as estimated.
  unestimatedTileConfigs.erase(config);

  // Clone a temporary loop band by cloning the outermost loop.
  auto outerLoop = band.front();
  auto tmpOuterLoop = outerLoop.clone();
  AffineLoopBand tmpBand;
  getLoopBandFromOutermost(tmpOuterLoop, tmpBand);

  // Insert the clone loop band to the front of the original band for the
  // convenience of the estimation.
  auto builder = OpBuilder(func);
  builder.setInsertionPoint(outerLoop);
  builder.insert(tmpOuterLoop);

  // Apply the tile config and estimate the loop band.
  auto tileList = getTileList(config);
  emitTileListDebugInfo(tileList);

  // Calculate the total iteration number.
  unsigned iterNum = 1;
  for (unsigned i = 0, e = tileList.size(); i < e; ++i)
    iterNum *= tripCountList[i] / tileList[i];

  // We always don't fully unroll all loops in the loop band.
  if (iterNum == 1)
    return false;

  // Apply the current tiling config and start the estimation. Note that after
  // optimization, tmpBand is optimized in place and becomes a new loop band.
  if (!applyOptStrategy(tmpBand, func, tileList, (unsigned)1))
    return false;
  tmpOuterLoop = tmpBand.front();
  estimator.estimateLoop(tmpOuterLoop, func);

  // Fetch latency and resource utilization.
  auto tmpInnerLoop = tmpBand.back();
  auto info = getLoopInfo(tmpInnerLoop);
  auto resource = getResource(tmpOuterLoop);
  assert(info && resource && "loop info or resource is not estimated");
  auto totalDsp = resource.getDsp() * info.getMinII();

  // Improve target II until II is equal to iteration latency. Note that when II
  // equal to iteration latency, the pipeline pragma is similar to a region
  // fully unroll pragma which unrolls all contained loops.
  for (auto tmpII = info.getMinII(); tmpII <= info.getIterLatency(); ++tmpII) {
    auto tmpDspNum = totalDsp / tmpII + 1;
    auto tmpLatency = info.getIterLatency() + tmpII * (iterNum - 1) + 2;
    auto point = LoopDesignPoint(tmpLatency, tmpDspNum, config, tmpII);

    allPoints.push_back(point);
    if (tmpDspNum <= maxDspNum)
      paretoPoints.push_back(point);
  }

  // Erase the temporary loop band.
  tmpOuterLoop.erase();
  return true;
}

/// Initialize the design space.
void LoopDesignSpace::initializeLoopDesignSpace(unsigned maxInitParallel) {
  LLVM_DEBUG(llvm::dbgs() << "Initialize the loop design space...\n";);

  for (TileConfig config = 0; config < validTileConfigNum; ++config) {
    auto tileList = getTileList(config);

    // We only evaluate the design points whose overall parallel is smaller
    // than the maxInitParallel to improve the efficiency.
    auto parallel = std::accumulate(tileList.begin(), tileList.end(),
                                    (unsigned)1, std::multiplies<unsigned>());

    if (parallel <= maxInitParallel) // || config == parallelConfig)
      evaluateTileConfig(config);
  }

  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  updateParetoPoints(paretoPoints);
}

/// Dump pareto and non-pareto points which have been evaluated in the design
/// space to a csv output file.
void LoopDesignSpace::dumpLoopDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.
  for (unsigned i = 0; i < tripCountList.size(); ++i)
    os << "l" << i << ",";
  os << "ii,cycle,dsp,type\n";

  // Print pareto design points.
  for (auto &point : paretoPoints) {
    for (auto size : getTileList(point.tileConfig))
      os << size << ",";
    os << point.targetII << "," << point.latency << "," << point.dspNum
       << ",pareto\n";
  }

  // Print all design points.
  for (auto &point : allPoints) {
    for (auto size : getTileList(point.tileConfig))
      os << size << ",";
    os << point.targetII << "," << point.latency << "," << point.dspNum
       << ",non-pareto\n";
  }

  csvFile->keep();
  LLVM_DEBUG(llvm::dbgs() << "Loop design space is dumped to file \""
                          << csvFilePath << "\".\n\n");
}

/// Get a random tile config which is one of the closest neighbors of "point".
Optional<TileConfig>
LoopDesignSpace::getRandomClosestNeighbor(LoopDesignPoint point,
                                          float maxDistance) {
  // Traverse all unestimated tile configs and collect all neighbors.
  SmallVector<std::pair<float, TileConfig>, 8> candidateConfigs;
  for (auto config : unestimatedTileConfigs) {
    auto distance = getTileConfigDistance(point.tileConfig, config);
    if (distance <= maxDistance)
      candidateConfigs.push_back(
          std::pair<float, TileConfig>(distance, config));
  }

  if (candidateConfigs.empty())
    return Optional<TileConfig>();

  // Sort candidate configs and collect the closest points.
  llvm::sort(candidateConfigs);
  SmallVector<TileConfig, 8> closestConfigs;
  float minDistance = maxDistance;

  for (auto configPair : candidateConfigs) {
    if (configPair.first <= minDistance) {
      closestConfigs.push_back(configPair.second);
      minDistance = configPair.first;
    } else
      break;
  }

  // Randomly pick one as the return point.
  std::srand(time(0));
  llvm::shuffle(closestConfigs.begin(), closestConfigs.end(),
                []() { return std::rand(); });

  return closestConfigs.front();
}

void LoopDesignSpace::exploreLoopDesignSpace(unsigned maxIterNum,
                                             float maxDistance) {
  LLVM_DEBUG(llvm::dbgs() << "Explore the loop design space...\n";);

  // Exploration loop of the dse.
  for (unsigned i = 0; i < maxIterNum; ++i) {
    std::srand(time(0));
    llvm::shuffle(paretoPoints.begin(), paretoPoints.end(),
                  []() { return std::rand(); });

    bool foundValidNeighbor = false;
    for (auto &point : paretoPoints) {
      if (!point.isActive)
        continue;

      auto closestNeighbor = getRandomClosestNeighbor(point, maxDistance);
      if (!closestNeighbor) {
        point.isActive = false;
        continue;
      }

      foundValidNeighbor = true;
      auto config = closestNeighbor.value();
      auto tileList = getTileList(config);

      evaluateTileConfig(config);
      break;
    }

    // Early termination if no valid neighbor is found.
    if (!foundValidNeighbor)
      break;

    // Update pareto points after each dse iteration.
    updateParetoPoints(paretoPoints);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n\n";);
}

//===----------------------------------------------------------------------===//
// FuncDesignSpace Class Definition
//===----------------------------------------------------------------------===//

void FuncDesignSpace::dumpFuncDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.
  for (unsigned i = 0, ei = loopDesignSpaces.size(); i < ei; ++i) {
    auto &loopSpace = loopDesignSpaces[i];

    for (unsigned j = 0, ej = loopSpace.tripCountList.size(); j < ej; ++j)
      os << "b" << i << "l" << j << ",";
    os << "b" << i << "ii,";
  }
  os << "cycle,dsp,type\n";

  // Print pareto design points.
  for (auto &funcPoint : paretoPoints) {
    for (unsigned i = 0, e = loopDesignSpaces.size(); i < e; ++i) {
      auto &loopPoint = funcPoint.loopDesignPoints[i];
      auto &loopSpace = loopDesignSpaces[i];

      for (auto size : loopSpace.getTileList(loopPoint.tileConfig))
        os << size << ",";
      os << loopPoint.targetII << ",";
    }
    os << funcPoint.latency << "," << funcPoint.dspNum << ",pareto\n";
  }

  csvFile->keep();
  LLVM_DEBUG(llvm::dbgs() << "Function design space is dumped to file \""
                          << csvFilePath << "\".\n\n");
}

void FuncDesignSpace::combLoopDesignSpaces() {
  LLVM_DEBUG(llvm::dbgs() << "Combine the loop design spaces...\n";);

  // Initialize the function design space with the first loop design space.
  auto &firstLoopSpace = loopDesignSpaces[0];
  for (auto &loopPoint : firstLoopSpace.paretoPoints) {
    // Annotate the first loop.
    auto loop = targetLoops[0];
    setTiming(loop, -1, -1, loopPoint.latency, -1);
    setResource(loop, -1, loopPoint.dspNum, -1);

    // Estimate the function and generate a new function design point.
    estimator.estimateFunc(func);
    auto latency = getTiming(func).getLatency();
    auto dspNum = getResource(func).getDsp();
    auto funcPoint = FuncDesignPoint(latency, dspNum, loopPoint);

    paretoPoints.push_back(funcPoint);
  }

  updateParetoPoints(paretoPoints);
  LLVM_DEBUG(llvm::dbgs() << "Iteration 0 pareto points number: "
                          << paretoPoints.size() << "\n";);

  // Combine other loop design spaces to the function design space one by one.
  for (unsigned i = 1, e = loopDesignSpaces.size(); i < e; ++i) {
    SmallVector<FuncDesignPoint, 16> newParetoPoints;
    auto &loopSpace = loopDesignSpaces[i];

    // Traverse all function design points.
    for (auto &funcPoint : paretoPoints) {
      // Annotate latency and dsp to all loops that are already included in the
      // function point, they are static for all design points of the new loop.
      for (unsigned ii = 0; ii < i; ++ii) {
        auto &oldLoopPoint = funcPoint.loopDesignPoints[ii];
        auto oldLoop = targetLoops[ii];
        setTiming(oldLoop, -1, -1, oldLoopPoint.latency, -1);
        setResource(oldLoop, -1, oldLoopPoint.dspNum, -1);
      }

      // Traverse all design points of the NEW loop.
      for (auto &loopPoint : loopSpace.paretoPoints) {
        // Annotate the new loop,
        auto loop = targetLoops[i];
        setTiming(loop, -1, -1, loopPoint.latency, -1);
        setResource(loop, -1, loopPoint.dspNum, -1);

        // Estimate the function and generate a new function design point.
        auto loopPoints = funcPoint.loopDesignPoints;
        loopPoints.push_back(loopPoint);

        estimator.estimateFunc(func);
        auto latency = getTiming(func).getLatency();
        auto dspNum = getResource(func).getDsp();
        auto newFuncPoint = FuncDesignPoint(latency, dspNum, loopPoints);

        newParetoPoints.push_back(newFuncPoint);
      }
    }

    // Update pareto points after each combination.
    updateParetoPoints(newParetoPoints);
    paretoPoints = newParetoPoints;
    LLVM_DEBUG(llvm::dbgs() << "Iteration " << i << " pareto points number: "
                            << paretoPoints.size() << "\n";);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n";);
}

bool FuncDesignSpace::exportParetoDesigns(unsigned outputNum,
                                          StringRef outputRootPath) {
  unsigned paretoNum = paretoPoints.size();
  auto sampleStep = std::max(paretoNum / outputNum, (unsigned)1);

  // Traverse all detected pareto points.
  unsigned sampleIndex = 0;
  for (auto &funcPoint : paretoPoints) {
    // Only export sampled points.
    if (sampleIndex % sampleStep == 0) {
      std::vector<FactorList> tileLists;
      SmallVector<unsigned, 4> targetIIs;

      for (unsigned i = 0; i < loopDesignSpaces.size(); ++i) {
        auto &loopSpace = loopDesignSpaces[i];
        auto &loopPoint = funcPoint.loopDesignPoints[i];
        auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
        auto targetII = loopPoint.targetII;

        tileLists.push_back(tileList);
        targetIIs.push_back(targetII);
      }

      // Clone a new function and apply optimization.
      auto tmpFunc = func.clone();
      if (!applyOptStrategy(tmpFunc, tileLists, targetIIs))
        return false;
      estimator.estimateFunc(tmpFunc);

      // Parse a new output file.
      auto outputFilePath = outputRootPath.str() + func.getName().str() +
                            "_pareto_" + std::to_string(sampleIndex) + ".mlir";

      std::string errorMessage;
      auto outputFile = mlir::openOutputFile(outputFilePath, &errorMessage);
      if (!outputFile)
        return false;

      auto &os = outputFile->os();
      os << tmpFunc << "\n";
      outputFile->keep();
    }
    ++sampleIndex;
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Sampled pareto points MLIR files are exported to path \""
                   << outputRootPath << "\".\n\n");
  return true;
}

//===----------------------------------------------------------------------===//
// Explorer Class Definition
//===----------------------------------------------------------------------===//

bool ScaleHLSExplorer::emitQoRDebugInfo(func::FuncOp func,
                                        std::string message) {
  estimator.estimateFunc(func);
  // auto latency = getTiming(func).getLatency();
  auto dspNum = getResource(func).getDsp();

  LLVM_DEBUG(llvm::dbgs() << message + "\n";
             //  llvm::dbgs() << "The clock cycle is " << Twine(latency)
             //               << ", DSP usage is " << Twine(dspNum) << ".\n\n";
  );

  return dspNum <= maxDspNum;
}

static int64_t getInnerParallelism(Block &block) {
  int64_t count = 0;
  for (auto loop : block.getOps<AffineForOp>()) {
    auto innerCount = getInnerParallelism(loop.getLoopBody().front());
    if (auto trip = getAverageTripCount(loop))
      count += trip.value() * innerCount;
    else
      count += innerCount;
  }

  // If the current loop is innermost loop, count should be one.
  return std::max(count, (int64_t)1);
}

bool ScaleHLSExplorer::evaluateFuncPipeline(func::FuncOp func) { return true; }

/// DSE Stage1: Simplify loop nests by unrolling. If we take the following loops
/// as example, where each nodes represents one sequential loop nests (LN). In
/// the simplification, we'll first try to pipeline LN1 and LN6. Suppose
/// unrolling LN6's region meets the resource constaints while pipelining LN1
/// not, we'll unroll LN6's region (fully unroll LN7 and LN8) and keep LN1
/// untouched. In the next step, we'll look into LN1 and check whether LN2's
/// region can be unrolled. Suppose unrolling LN2's region meets the resource
/// constraints, we'll unrolling LN2's region (fully unroll LN7 and LN8). Note
/// that in this simplification, all LNs that don't contain any LNs will not be
/// touched, such as LN5. Their optimization will be explored later. This
/// procedure will recursively applied to inner LNs until no eligible LN exists.
///
///     LN1      LN6
///      |        |
///     / \      / \
///   LN2 LN5  LN7 LN8
///    |
///   / \
/// LN3 LN4
///
/// After the simplification, the loop becomes the following one, where LN1 has
/// been proved untouchable as region loop unrolling is the primary optimization
/// that consumes the least extra resources. Formally, in the simplified
/// function, all non-leaf LNs is untouchable (LN1) and only leaf LNs can be
/// further optimized (LN2, LN5, and LN6).
///
///     LN1      LN6
///      |
///     / \
///   LN2 LN5
///
/// TODO: there is a large design space in this simplification.
bool ScaleHLSExplorer::simplifyLoopNests(func::FuncOp func) {
  LLVM_DEBUG(llvm::dbgs()
                 << "----------\nStage1: Simplify loop nests structure...\n";);

  auto funcForOps = func.getOps<AffineForOp>();
  auto targetLoops =
      SmallVector<AffineForOp, 8>(funcForOps.begin(), funcForOps.end());

  while (!targetLoops.empty()) {
    SmallVector<std::pair<int64_t, AffineForOp>, 8> candidateLoops;

    // Collect all candidate loops. Here, only loops whose innermost loop has
    // more than one inner loops will be considered as a candidate.
    for (auto target : targetLoops) {
      AffineLoopBand loopBand;
      auto innermostLoop = getLoopBandFromOutermost(target, loopBand);

      // Calculate the overall introduced parallelism if the innermost loop of
      // the current loop band is fully unrolled.
      auto parallelism =
          getInnerParallelism(innermostLoop.getLoopBody().front());

      // Collect all candidate loops into an vector, we'll ignore too large
      // parallelism as unrolling them typically introduce very high cost.
      if (parallelism > 1 && parallelism <= 512)
        candidateLoops.push_back(
            std::pair<int64_t, AffineForOp>(parallelism, innermostLoop));
    }

    // Since all target loops have been handled, clear the targetLoops vector.
    targetLoops.clear();

    // Sort the candidate loops.
    llvm::sort(candidateLoops);

    // Traverse all candidates to check whether applying fully loop unrolling
    // has violation with the resource constraints. If so, add all inner loops
    // into targetLoops. Otherwise, fully unroll the candidate.
    for (auto pair : candidateLoops) {
      auto candidate = pair.second;

      // Create a temporary function.
      candidate->setAttr("opt_flag", BoolAttr::get(func.getContext(), true));
      auto tmpFunc = func.clone();

      // Find the candidate loop in the temporary function and apply fully loop
      // unrolling to it.
      tmpFunc.walk([&](AffineForOp loop) {
        if (loop->getAttrOfType<BoolAttr>("opt_flag")) {
          applyFullyLoopUnrolling(*loop.getBody());
          applyMemoryOpts(tmpFunc);
          applyAutoArrayPartition(tmpFunc);
          return;
        }
      });

      // Estimate the temporary function.
      estimator.estimateFunc(tmpFunc);

      // Fully unroll the candidate loop or delve into child loops.
      if (getResource(tmpFunc).getDsp() <= maxDspNum) {
        applyFullyLoopUnrolling(*candidate.getBody());
        applyMemoryOpts(func);
        applyAutoArrayPartition(func);
      } else {
        auto childForOps = candidate.getOps<AffineForOp>();
        targetLoops.append(childForOps.begin(), childForOps.end());
      }

      candidate->removeAttr("opt_flag");
    }
  }

  return emitQoRDebugInfo(func, "\nFinish Stage1.");
}

/// DSE Stage2: Optimize leaf loop nests. Different optimization conbinations
/// will be applied to each leaf LNs, and the best one which meets the resource
/// constraints will be picked as the final solution.
/// TODO: better handle variable bound kernels.
bool ScaleHLSExplorer::optimizeLoopBands(func::FuncOp func,
                                         bool directiveOnly) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage2: Apply loop perfection, loop "
                             "order opt, and remove variable loop bound...\n";);

  AffineLoopBands targetBands;
  getLoopBands(func.front(), targetBands);
  unsigned targetNum = targetBands.size();

  // Loop perfection, remove variable bound, and loop order optimization are
  // always applied for the convenience of polyhedral optimizations.
  for (unsigned i = 0; i < targetNum; ++i) {
    auto &band = targetBands[i];
    LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    applyAffineLoopPerfection(band);

    // If only explore directive optimizations, disable loop oreder opt.
    if (!directiveOnly)
      applyAffineLoopOrderOpt(band);

    applyRemoveVariableBound(band);
  }

  return emitQoRDebugInfo(func, "\nFinish Stage2.");
}

class CountingSemaphore {
public:
  explicit CountingSemaphore(int permits) : permits_(permits) {}

  void acquire() {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [&] { return permits_ > 0; });
    --permits_;
  }

  void release() {
    std::lock_guard<std::mutex> lock(m_);
    ++permits_;
    cv_.notify_one();
  }

private:
  std::mutex m_;
  std::condition_variable cv_;
  int permits_;
};

// If your project already typedefs this, remove this line.
using AffineLoopBand = llvm::SmallVector<mlir::AffineForOp, 6>;

struct SpaceItem {
  LoopDesignSpace space;
  unsigned origBandIdx;
};


// Final function to explore the design space of loop bands in parallel.
FuncDesignSpace exploreLoopBandsInParallelV2(mlir::func::FuncOp func,
                                             ArrayRef<AffineLoopBand> targetBands,
                                             ScaleHLSEstimator &estimator,
                                             unsigned maxDspNum,
                                             unsigned maxExplParallel,
                                             unsigned maxLoopParallel,
                                             bool directiveOnly,
                                             unsigned maxInitParallel,
                                             unsigned maxIterNum,
                                             float maxDistance,
                                             llvm::StringRef csvRootPath,
                                             unsigned maxParallel, 
                                             llvm::StringMap<int64_t> latencyMap,
                                             llvm::StringMap<int64_t> dspMap)
{
  using AffineLoopBand = llvm::SmallVector<mlir::AffineForOp, 6>;
  // Getting the csvRootPath and function name.
  const std::string root = csvRootPath.str();
  const std::string name = func.getName().str();

  const size_t N = targetBands.size();

   // Build an empty SmallVector and construct FuncDesignSpace with it.
  llvm::SmallVector<LoopDesignSpace, 4> initial;
  initial.reserve(N);                           
  FuncDesignSpace funcDesignSpace(func, initial, estimator, maxDspNum);
  funcDesignSpace.loopDesignSpaces.reserve(N);

  //Cloning the module once per band so each thread has its own IR
  auto parent = func->getParentOfType<mlir::ModuleOp>();
  if (!parent || targetBands.empty()) {
    auto emptyBase = func.clone();
    llvm::SmallVector<LoopDesignSpace, 4> empty;
    return FuncDesignSpace(emptyBase, empty, estimator, maxDspNum);
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modClones(N);
  std::vector<mlir::func::FuncOp> fClones(N);
  std::vector<AffineLoopBand> bandClones(N);

  for (size_t i = 0; i < N; ++i) {
    modClones[i] = mlir::cast<mlir::ModuleOp>(parent.clone());
    auto f = modClones[i]->lookupSymbol<mlir::func::FuncOp>(name);
    fClones[i] = f;
    mlir::scalehls::AffineLoopBands bandsInClone;
    getLoopBands(f.front(), bandsInClone);
    bandClones[i] = bandsInClone[i];
  }

  llvm::SmallVector<LoopDesignSpace, 4> loopSpaces;
  llvm::SmallVector<unsigned, 4> loopSpaceOrigIdx;
  loopSpaces.reserve(N);
  loopSpaceOrigIdx.reserve(N);

  // Keeping one estimator for a single thread.
  std::vector<std::unique_ptr<ScaleHLSEstimator>> estimators;
  estimators.reserve(N);

  for (size_t i = 0; i < N; ++i){
    estimators.emplace_back(std::make_unique<ScaleHLSEstimator>(
      latencyMap, dspMap, true));
    loopSpaces.emplace_back(fClones[i], bandClones[i], *estimators.back(),
                            maxDspNum, maxExplParallel, maxLoopParallel, directiveOnly);
    loopSpaceOrigIdx.push_back(i);
  }

  // for (size_t i = 0; i < N; ++i) {
  //   loopSpaces.emplace_back(fClones[i], bandClones[i], *estimators[i],
  //                           maxDspNum, maxExplParallel, maxLoopParallel, directiveOnly);
  //   loopSpaceOrigIdx.push_back(i);
  // }

  std::vector<std::future<void>> fut;
  fut.reserve(N);
  unsigned hw = std::thread::hardware_concurrency();
  unsigned lim = maxParallel ? maxParallel : (hw ? hw : 4);
  std::atomic<unsigned> tickets{lim};

  for (size_t i = 0; i < N; ++i) {
    fut.emplace_back(std::async(std::launch::async, [&, i] {
      llvm::errs() << "[PARELLEL] about to run V2, i='" << i << "'\n";
      // naive ticketing to throttle
      while (tickets.load() == 0) std::this_thread::yield();
      tickets.fetch_sub(1);

      auto &space = loopSpaces[i];
      space.initializeLoopDesignSpace(maxInitParallel);
      space.exploreLoopDesignSpace(maxIterNum, maxDistance);
      // unique CSV per band
      std::string csv = root + name + "_loop_" + std::to_string(i) + "_space.csv";
      space.dumpLoopDesignSpace(csv);

      tickets.fetch_add(1);
      llvm::errs() << "[PARELLEL] Finished run V2 for , i='" << i << "'\n";
    }));
  }

  for (auto &f : fut) f.get();

  
  llvm::SmallVector<LoopDesignSpace, 4> filtered;
  llvm::SmallVector<unsigned, 4> filteredOrigIdx;
  filtered.reserve(loopSpaces.size());
  filteredOrigIdx.reserve(loopSpaces.size());
  for (size_t i = 0; i < loopSpaces.size(); ++i) {
    auto &ls = loopSpaces[i];
    if (!ls.paretoPoints.empty()) {
      filtered.push_back(std::move(ls));
      filteredOrigIdx.push_back(loopSpaceOrigIdx[i]);
    }
  }
  for (auto &ls : loopSpaces) {
    if (!ls.paretoPoints.empty()) {
      filtered.push_back(std::move(ls));
    }
    else {
      llvm::errs() << "[DSE] Band produced 0 points, dropping\n";
    }
  }

  funcDesignSpace.loopDesignSpaces.clear();
  funcDesignSpace.loopDesignSpaces.reserve(filtered.size());
  funcDesignSpace.bandIndexMap = {};                                     // make this a member: SmallVector<unsigned,4>
  funcDesignSpace.bandIndexMap.reserve(filteredOrigIdx.size());
  for (size_t i = 0; i < filtered.size(); ++i) {
    funcDesignSpace.loopDesignSpaces.push_back(std::move(filtered[i]));
    funcDesignSpace.bandIndexMap.push_back(filteredOrigIdx[i]);  
  }
  
  funcDesignSpace.combLoopDesignSpaces();                          

  // Dump design points to csv file for each function.
  auto funcCsvFilePath =
      csvRootPath.str() + func.getName().str() + "_space.csv";
  funcDesignSpace.dumpFuncDesignSpace(funcCsvFilePath);

  // Apply the best function design point under the constraints.
  for (auto &funcPoint : funcDesignSpace.paretoPoints) {
    if (funcPoint.dspNum <= maxDspNum) {
      std::vector<FactorList> tileLists;
      SmallVector<unsigned, 4> targetIIs;

      // for (unsigned i = 0; i < N; ++i) {
      //   auto &loopSpace = funcDesignSpace.loopDesignSpaces[i];
      //   auto &loopPoint = funcPoint.loopDesignPoints[i];
      //   auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
      //   auto targetII = loopPoint.targetII;

      //   llvm::errs() << "Loop band " << i << ": "
      //                           << "Loop tiling & pipelining config ( ";
      //   for (auto tile : tileList) { llvm::errs() << tile << ","; };
      //   llvm::errs() << targetII << ")\n";

      //   tileLists.push_back(tileList);
      //   targetIIs.push_back(targetII);
      // }
      mlir::scalehls::AffineLoopBands bandsNow;
      getLoopBands(func.front(), bandsNow);
      tileLists.reserve(funcDesignSpace.loopDesignSpaces.size());
      targetIIs.reserve(funcDesignSpace.loopDesignSpaces.size());

      for (size_t outIdx = 0; outIdx < funcDesignSpace.loopDesignSpaces.size(); ++outIdx) {
        unsigned origIdx = funcDesignSpace.bandIndexMap[outIdx];

        auto &loopSpace = funcDesignSpace.loopDesignSpaces[outIdx];
        auto &loopPoint = funcPoint.loopDesignPoints[origIdx];

        auto tiles = loopSpace.getTileList(loopPoint.tileConfig);

        // Defensive: make tiles match the band loop count
        const size_t want = bandsNow[outIdx].size();
        if (tiles.size() > want) {
          // drop extras (e.g., accidental II)
          llvm::errs() << "Dropping extra tiles from band " << outIdx << "\n";
          tiles.resize(want);       
        }       
        if (tiles.size() < want) tiles.resize(want, 1);

        tileLists.push_back(std::move(tiles));
        targetIIs.push_back(loopPoint.targetII);
      }
      if (!applyOptStrategy(func, tileLists, targetIIs))
      break;
    }
  }

  return funcDesignSpace;
}

/// DSE Stage3: Explore the function design space through dynamic programming.
bool ScaleHLSExplorer::exploreDesignSpace(func::FuncOp func, bool directiveOnly,
                                          StringRef outputRootPath,
                                          StringRef csvRootPath, 
                                          llvm::StringMap<int64_t> latencyMap,
                                          llvm::StringMap<int64_t> dspMap) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage3: Conduct top function design "
                             "space exploration...\n";);

  auto tmpFunc = func.clone();
  AffineLoopBands targetBands;
  getLoopBands(tmpFunc.front(), targetBands);
  unsigned targetNum = targetBands.size();
  using AffineLoopBand = llvm::SmallVector<mlir::AffineForOp, 6>;
  std::vector<AffineLoopBand> bandStore;
  bandStore.reserve(targetBands.size());
  for (const auto &b : targetBands) bandStore.push_back(b);

  std::cerr << "Before calling the DSE with csvRootPath as " << csvRootPath.str() << std::endl;

  llvm::errs() << "[CALL] about to run V2, csvRoot='" << csvRootPath << "'\n";
  auto funcSpace = exploreLoopBandsInParallelV2(func, targetBands, estimator,
                                                maxDspNum, maxExplParallel, maxLoopParallel,
                                                directiveOnly, maxInitParallel, maxIterNum,
                                                maxDistance, csvRootPath, 0, latencyMap, dspMap);
  llvm::errs() << "[CALL] V2 finished for " << tmpFunc.getName() << "\n";
 

  return emitQoRDebugInfo(func, "\nFinish Stage3.");
}

//===----------------------------------------------------------------------===//
// DesignSpaceExplore Entry
//===----------------------------------------------------------------------===//

/// This is a temporary approach that does not scale.
void ScaleHLSExplorer::applyDesignSpaceExplore(func::FuncOp func,
                                               bool directiveOnly,
                                               StringRef outputRootPath,
                                               StringRef csvRootPath, 
                                               llvm::StringMap<int64_t> latencyMap, 
                                               llvm::StringMap<int64_t> dspMap) {
  emitQoRDebugInfo(func, "Start multiple level DSE.");

  // Simplify loop nests by unrolling.
  if (!simplifyLoopNests(func))
    return;

  // Optimize loop bands by loop perfection, loop order permutation, and loop
  // rectangularization.
  if (!optimizeLoopBands(func, directiveOnly))
    return;

  // Explore the design space through a multiple level approach.
  if (!exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, latencyMap, dspMap))
    return;
}


// Helpers for the Parellel DSE run.

static inline std::string sanitizeName(std::string s) {
  for (char& c : s) if (!(std::isalnum((unsigned char)c) || c=='_' || c=='-')) c = '_';
  return s;
}


template <typename ExplorerT>
void runDSEInParallel(mlir::ModuleOp module,
                      ExplorerT &explorer,
                      bool directiveOnly,
                      const std::filesystem::path &outRoot,
                      const std::filesystem::path &csvRoot,
                      int maxParallel,
                      bool clonePerKernel, 
                      llvm::StringMap<int64_t> latencyMap, 
                      llvm::StringMap<int64_t> dspMap) {

  std::cerr << "Running DSE in parallel..." << std::endl;
  namespace fs = std::filesystem;
  fs::create_directories(outRoot);
  fs::create_directories(csvRoot);

  if (maxParallel <= 0) {
    unsigned hc = std::thread::hardware_concurrency();
    // Having one task per core.
    maxParallel = (hc == 0) ? 2 : (int)hc;
    std::cerr << "Running DSE in parallel... with " << maxParallel << " threads." << std::endl;
  }

  // Bounded parallelism across kernels
  CountingSemaphore gate(maxParallel);

  std::vector<std::future<void>> futures;
  futures.reserve(std::distance(module.getOps<mlir::func::FuncOp>().begin(),
                                module.getOps<mlir::func::FuncOp>().end()));

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    // Only top functions are kernels
    auto attrs = func->getAttrDictionary();
    if (!attrs.contains("top_func"))
      continue;

    // Acquire a slot before launching
    gate.acquire();


    // Capture only what you need by value
    auto funcName = sanitizeName(func.getName().str());
    fs::path kernelOutDir = outRoot;
    fs::path kernelCsv    = csvRoot;
    // fs::path kernelOutDir = outRoot/ funcName;
    // fs::path kernelCsv    = csvRoot/ (funcName + ".csv");

    futures.emplace_back(std::async(std::launch::async, [=, &explorer, &gate]() {
      std::error_code ec;
        std::filesystem::create_directories(kernelOutDir, ec);

        const auto tid = llvm::get_threadid();
        std::cerr << "[THREAD " << tid << "] || Starting DSE for kernel: "
                << funcName << " | Output: " << kernelOutDir << std::endl;

        llvm::errs() << "[DBG] func=" << funcName << "\n";
        llvm::errs() << "[DBG] outDir=" << kernelOutDir.string() << "\n";
        llvm::errs() << "[DBG] csv=" << kernelCsv.string() << "\n";


        // Now calling the function to be running in parellel.
       explorer.applyDesignSpaceExplore(func,
                                 directiveOnly,
                                 kernelOutDir.string(),
                                 kernelCsv.string(), 
                                 latencyMap,
                                 dspMap);

        std::cerr << "[THREAD " << tid << "] || Finished DSE for kernel: "
                << funcName << " | Output: " << kernelOutDir << std::endl;

      gate.release();
    }));
  }

  for (auto &f : futures) f.get();

  // mergeKernelCSVs(csvRoot, csvRoot / "merged.csv");
}

void mergeKernelCSVs(const std::filesystem::path& csvRoot, const std::filesystem::path& mergedCsvPath) {
  std::ofstream out(mergedCsvPath);
  if (!out) return;

  bool wroteHeader = false;
  for (auto& entry : std::filesystem::directory_iterator(csvRoot)) {
    if (!entry.is_regular_file()) continue;
    const auto& p = entry.path();
    if (p.extension() != ".csv") continue;

    std::ifstream in(p);
    if (!in) continue;

    std::string line;
    std::string kernel = p.stem().string(); // filename without .csv

    // Read header
    if (std::getline(in, line)) {
      // First file: write merged header with "Kernel" column
      if (!wroteHeader) {
        out << "Kernel," << line << '\n';
        wroteHeader = true;
      }
      // For subsequent files, skip their header if it matches (simple heuristic)
      else if (line.find(',') == std::string::npos || line.find("Latency") == std::string::npos) {
        // header looks odd; treat it as data
        out << kernel << ',' << line << '\n';
      }
    }
    // Write data lines
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      out << kernel << ',' << line << '\n';
    }
  }
  out.flush();
}

namespace {
struct DesignSpaceExplore : public DesignSpaceExploreBase<DesignSpaceExplore> {
  DesignSpaceExplore() = default;
  DesignSpaceExplore(std::string dseTargetSpec) { targetSpec = dseTargetSpec; }

  void runOnOperation() override {
    auto module = getOperation();

    // Read target specification JSON file.
    std::string errorMessage;
    auto configFile = mlir::openInputFile(targetSpec, &errorMessage);
    if (!configFile) {
      llvm::errs() << errorMessage << "\n";
      return signalPassFailure();
    }

    // Parse JSON file into memory.
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) {
      llvm::errs() << "failed to parse the target spec json file\n";
      return signalPassFailure();
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) {
      llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
      return signalPassFailure();
    }

    // Collect DSE configurations.
    unsigned outputNum = configObj->getInteger("output_num").value_or(30);

    unsigned maxInitParallel =
        configObj->getInteger("max_init_parallel").value_or(32);
    unsigned maxExplParallel =
        configObj->getInteger("max_expl_parallel").value_or(1024);
    unsigned maxLoopParallel =
        configObj->getInteger("max_loop_parallel").value_or(128);

    assert(maxInitParallel <= maxExplParallel &&
           maxLoopParallel <= maxExplParallel &&
           "invalid configuration of DSE");

    unsigned maxIterNum = configObj->getInteger("max_iter_num").value_or(30);
    float maxDistance = configObj->getNumber("max_distance").value_or(3.0);

    bool directiveOnly =
        configObj->getBoolean("directive_only").value_or(false);
    bool resourceConstr =
        configObj->getBoolean("resource_constr").value_or(true);

    // Collect profiling latency and DSP usage data, where default values are
    // based on Xilinx PYNQ-Z1 board.
    llvm::StringMap<int64_t> latencyMap;
    getLatencyMap(configObj, latencyMap);
    llvm::StringMap<int64_t> dspUsageMap;
    getDspUsageMap(configObj, dspUsageMap);

    unsigned maxDspNum = ceil(configObj->getInteger("dsp").value_or(220) * 1.1);
    if (!resourceConstr)
      maxDspNum = UINT_MAX;

    // Initialize an performance and resource estimator.
    auto estimator = ScaleHLSEstimator(latencyMap, dspUsageMap, true);
    auto explorer = ScaleHLSExplorer(estimator, outputNum, maxDspNum,
                                     maxInitParallel, maxExplParallel,
                                     maxLoopParallel, maxIterNum, maxDistance);

    // Optimize the top function.
    // TODO: Support to contain sub-functions.
    // for (auto func : module.getOps<func::FuncOp>()) {
    //   if (hasTopFuncAttr(func))
    //     explorer.applyDesignSpaceExplore(func, directiveOnly, outputPath,
    //                                      csvPath);
    // }

    runDSEInParallel(module, explorer,
          directiveOnly,
          std::filesystem::path(outputPath.c_str()),
          std::filesystem::path(csvPath.c_str()),
          0,      // auto cores
          false, 
          latencyMap, 
          dspUsageMap); // no cloning
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createDesignSpaceExplorePass(std::string dseTargetSpec) {
  return std::make_unique<DesignSpaceExplore>(dseTargetSpec);
}
