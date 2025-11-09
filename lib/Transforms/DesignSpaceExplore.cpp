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
// #include <pthread.h>

#define DEBUG_TYPE "scalehls"

using namespace mlir;
using namespace scalehls;

/// Update paretoPoints to remove design points that are not pareto frontiers.
/// Optionally filter by resource constraints.
template <typename DesignPointType>
static void updateParetoPoints(SmallVector<DesignPointType, 16> &paretoPoints,
                               unsigned maxDspNum = UINT_MAX,
                               unsigned maxBramNum = UINT_MAX) {
  LLVM_DEBUG(llvm::dbgs() << "Updating pareto points with maxDspNum: " << maxDspNum << " and maxBramNum: " << maxBramNum << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "Number of pareto points before filtering: " << paretoPoints.size() << "\n";);
  // First, filter by resource constraints if provided
  if (maxDspNum != UINT_MAX || maxBramNum != UINT_MAX) {
    SmallVector<DesignPointType, 16> filteredPoints;
    for (auto &point : paretoPoints) {
      bool withinConstraints = true;
      LLVM_DEBUG(llvm::dbgs() << "Checking point with dspNum: " << point.dspNum << " and maxDspNum: " << maxDspNum << "\n";);
      if (maxDspNum != UINT_MAX && point.dspNum > maxDspNum) {
        withinConstraints = false;
      }
      // Note: DesignPointType only has dspNum, not bramNum
      // BRAM checking would need to be done separately where we have the resource object
      if (withinConstraints) {
        filteredPoints.push_back(point);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Number of pareto points in filtered points: " << filteredPoints.size() << "\n";);
    paretoPoints = filteredPoints;
  }

  LLVM_DEBUG(llvm::dbgs() << "Number of pareto points after filtering: " << paretoPoints.size() << "\n";);

  if (paretoPoints.empty())
    return;

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

    } else if ((tmpDspNum == paretoDspNum && tmpLatency == paretoLatency) || frontiers.empty())
      frontiers.push_back(point);
  }

  LLVM_DEBUG(llvm::dbgs() << "updated pareto points, number of points: " << frontiers.size() << "\n";);

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
  updateParetoPoints(paretoPoints, maxDspNum);
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
    updateParetoPoints(paretoPoints, maxDspNum);
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

  // Check if there are any loop design spaces to combine.
  if (loopDesignSpaces.empty()) {
    llvm::errs() << "[DSE] WARNING: No loop bands found in function '" 
                 << func.getName() << "'. Skipping design space exploration.\n";
    LLVM_DEBUG(llvm::dbgs() << "No loop design spaces to combine.\n";);
    return;
  }

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
  // If no loop design points are found, create a default function design point.
  if (paretoPoints.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No loop design points found, creating a default function design point.\n";);
    estimator.estimateFunc(func);
    auto latency = getTiming(func).getLatency();
    auto dspNum = getResource(func).getDsp();
    auto funcPoint = FuncDesignPoint(latency, dspNum);
    paretoPoints.push_back(funcPoint);
  }


  updateParetoPoints(paretoPoints, maxDspNum);
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
    updateParetoPoints(newParetoPoints, maxDspNum);
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
// HierFuncDesignSpace Class Definition
//===----------------------------------------------------------------------===//

func::FuncOp HierFuncDesignSpace::getSubFunc(func::FuncOp func, StringRef subFuncName) {
  auto subFuncNameAttr = StringAttr::get(func.getContext(), subFuncName);
  auto subFuncActual = SymbolTable::lookupNearestSymbolFrom(func, subFuncNameAttr);
  if (!subFuncActual) {
    llvm::errs() << "[DSE] ERROR: Cannot find sub function '"
                 << subFuncName << "' in function '"
                 << func.getName() << "'\n";
    assert(false && "Cannot find sub function");
  }
  return dyn_cast<func::FuncOp>(subFuncActual);
}

void HierFuncDesignSpace::combFuncDesignSpaces() {
  LLVM_DEBUG(llvm::dbgs() << "Combine the function design spaces...\n";);

  // Check if there are any function design spaces to combine.
  if (subHierFuncDesignSpaces.empty()) {
    llvm::errs() << "[DSE] WARNING: No sub function hierarchical design spaces to combine in function '" 
                 << func.getName() << "'. Skipping design space exploration.\n";
    LLVM_DEBUG(llvm::dbgs() << "No sub function hierarchical design spaces to combine.\n";);
    return;
  }

  // Initialize the hierarchical function design space with the first function design space.
  for (auto &funcPoint : funcDesignSpace.paretoPoints) {

    auto currentFunc = funcDesignSpace.func;
    setTiming(currentFunc, -1, -1, funcPoint.latency, -1);
    setResource(currentFunc, -1, funcPoint.dspNum, -1);

    estimator.estimateFunc(currentFunc);
    auto latency = getTiming(currentFunc).getLatency();
    auto dspNum = getResource(currentFunc).getDsp();
    auto currentHierFuncPoint = HierFuncDesignPoint(latency, dspNum, funcPoint);
    paretoPoints.push_back(currentHierFuncPoint);
  }

  updateParetoPoints(paretoPoints, maxDspNum);
  LLVM_DEBUG(llvm::dbgs() << "Iteration 0 pareto points number: "
                          << paretoPoints.size() << "\n";);

  // Combine other hierarchical function design spaces to the current hierarchical function design space one by one.
  for (unsigned i = 0, e = subHierFuncDesignSpaces.size(); i < e; ++i) {
    SmallVector<HierFuncDesignPoint, 16> newParetoPoints;
    auto &subHierFuncSpace = subHierFuncDesignSpaces[i];

    // Traverse all hierarchical function design points.
    for (auto &hierFuncPoint : paretoPoints) {
      // Annotate latency and dsp to all functions that are already included in the
      // hierarchical function point, they are static for all design points of the new function.
      for (unsigned ii = 0; ii < i; ++ii) {
        auto &oldSubFuncPoint = hierFuncPoint.subHierFuncDesignPoints[ii];
        auto &oldSubHierFuncSpace = subHierFuncDesignSpaces[ii];
        auto oldSubFunc = oldSubHierFuncSpace.func;
        setTiming(oldSubFunc, -1, -1, oldSubFuncPoint.latency, -1);
        setResource(oldSubFunc, -1, oldSubFuncPoint.dspNum, -1);
      }

      // Traverse all design points of the next hierarchical function.
      for (auto &subHierFuncPoint : subHierFuncSpace.paretoPoints) {
        // Annotate the next hierarchical function.
        auto subFunc = getSubFunc(func, subHierFuncSpace.func.getName());
        setTiming(subFunc, -1, -1, subHierFuncPoint.latency, -1);
        setResource(subFunc, -1, subHierFuncPoint.dspNum, -1);

        // Estimate the top-level function and generate a new hierarchical function design point.
        auto subHierFuncPoints = hierFuncPoint.subHierFuncDesignPoints;
        subHierFuncPoints.push_back(subHierFuncPoint);

        estimator.estimateFunc(func);
        auto latency = getTiming(func).getLatency();
        auto dspNum = getResource(func).getDsp();
        auto newHierFuncPoint = HierFuncDesignPoint(latency, dspNum, hierFuncPoint.funcDesignPoint, subHierFuncPoints);
        newParetoPoints.push_back(newHierFuncPoint);
      }
    }

    // Update pareto points after each combination, filtering by resources.
    updateParetoPoints(newParetoPoints, maxDspNum);
    paretoPoints = newParetoPoints;
    LLVM_DEBUG(llvm::dbgs() << "Iteration " << i << " pareto points number: "
                            << paretoPoints.size() << "\n";);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n";);
}

void HierFuncDesignSpace::dumpHierFuncDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.

  // track current function design space
  for (unsigned i = 0; i < funcDesignSpace.loopDesignSpaces.size(); ++i) {
    auto &loopDesignSpace = funcDesignSpace.loopDesignSpaces[i];
    for (unsigned j = 0; j < loopDesignSpace.tripCountList.size(); ++j)
      os << "tf" << 0 << "b" << i << "l" << j << ",";
    os << "tf" << 0 << "b" << i << "ii,";
  }
  os << "tf_cycle,tf_dsp,";
  for (unsigned h = 0; h < subHierFuncDesignSpaces.size(); ++h) {
    os << "sf" << h << "cycle" << ",";
    os << "sf" << h << "dsp" << ",";
  }
  os << "cycle,dsp,type\n";

  // Print pareto design points.

  // start with current function space (loops, not hierarchical)
  for (auto &hierFuncPoint : paretoPoints) {
    auto &funcPoint = hierFuncPoint.funcDesignPoint;
    for (unsigned i = 0, e = funcDesignSpace.loopDesignSpaces.size(); i < e; ++i) {
      auto &loopPoint = funcPoint.loopDesignPoints[i];
      auto &loopSpace = funcDesignSpace.loopDesignSpaces[i];

      for (auto size : loopSpace.getTileList(loopPoint.tileConfig))
        os << size << ",";
      os << loopPoint.targetII << ",";
    }
    os << funcPoint.latency << "," << funcPoint.dspNum << ",";

    for (unsigned i = 0; i < subHierFuncDesignSpaces.size(); ++i) {
      auto &subHierFuncPoint = hierFuncPoint.subHierFuncDesignPoints[i];
      os << subHierFuncPoint.latency << "," << subHierFuncPoint.dspNum << ",";
    }
    os << hierFuncPoint.latency << "," << hierFuncPoint.dspNum << "," << "pareto\n";
  }

  csvFile->keep();
  LLVM_DEBUG(llvm::dbgs() << "Hierarchical function design space is dumped to file \""
                          << csvFilePath << "\".\n\n");
}

bool HierFuncDesignSpace::applyOptStrategyRecursive(func::FuncOp currentFunc, HierFuncDesignPoint hierFuncPoint) {
  LLVM_DEBUG(llvm::dbgs() << "Apply optimization strategies to the current function '"
                          << currentFunc.getName() << "'...\n";);
  // STEP 1: Apply optimization strategies to the current function
  auto funcPoint = hierFuncPoint.funcDesignPoint;
  std::vector<FactorList> tileLists;
  SmallVector<unsigned, 4> targetIIs;
  for (unsigned i = 0, e = funcDesignSpace.loopDesignSpaces.size(); i < e; ++i) {
    auto &loopPoint = funcPoint.loopDesignPoints[i];
    auto &loopSpace = funcDesignSpace.loopDesignSpaces[i];
    auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
    auto targetII = loopPoint.targetII;
    tileLists.push_back(tileList);
    targetIIs.push_back(targetII);
  }
  if (!applyOptStrategy(currentFunc, tileLists, targetIIs)){
    llvm::errs() << "[DSE] ERROR: Failed to apply optimization strategies to the current function '"
                 << currentFunc.getName() << "'\n";
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "Optimization strategies applied to the current function '"
                          << currentFunc.getName() << "'.\n";);
  // STEP 2: Apply optimization strategies to the sub functions
  for (unsigned i = 0; i < hierFuncPoint.subHierFuncDesignPoints.size(); ++i) {
    auto &subHierFuncPoint = hierFuncPoint.subHierFuncDesignPoints[i];
    auto &subHierFuncSpace = subHierFuncDesignSpaces[i];
    auto subFunc = getSubFunc(currentFunc, subHierFuncSpace.func.getName());
    if (!subHierFuncSpace.applyOptStrategyRecursive(subFunc, subHierFuncPoint))
      return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "Optimization strategies applied to the sub functions of '"
                          << currentFunc.getName() << "'.\n";);
  return true;
}

bool HierFuncDesignSpace::exportParetoDesigns(unsigned outputNum,
                                              StringRef outputRootPath) {
  unsigned paretoNum = paretoPoints.size();
  auto sampleStep = std::max(paretoNum / outputNum, (unsigned)1);

  // Traverse all detected pareto points.
  unsigned sampleIndex = 0;
  for (auto &hierFuncPoint : paretoPoints) {
    // Only export sampled points.
    if (sampleIndex % sampleStep == 0) {
      auto tmpFunc = func.clone();

      if (!applyOptStrategyRecursive(tmpFunc, hierFuncPoint))
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

HierFuncDesignSpace ScaleHLSExplorer::exploreHierDesignSpace(func::FuncOp func, bool directiveOnly,
                                              StringRef outputRootPath,
                                              StringRef csvRootPath) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage3: Conduct hierarchical function "
                             "design space exploration...\n";);

  // STEP 1: Create current function design space.
  FuncDesignSpace funcSpace = exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath);

  // STEP 2: Recursively create hierarchical function design spaces for sub functions.
  std::vector<HierFuncDesignSpace> subHierFuncDesignSpaces;
  for (auto subFunc : func.getOps<func::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "Exploring hierarchical function design space for sub function '"
                            << subFunc.getName() << "'...\n";);
    auto subHierFuncSpace = exploreHierDesignSpace(subFunc, directiveOnly, outputRootPath, csvRootPath);
    subHierFuncDesignSpaces.push_back(subHierFuncSpace);
  }
  LLVM_DEBUG(llvm::dbgs() << "Hierarchical function design spaces for sub functions created.\n";);
  // STEP 3: Combine function design spaces into a hierarchical function design space, including the current function design space.
  HierFuncDesignSpace hierFuncSpace = HierFuncDesignSpace(func, funcSpace, subHierFuncDesignSpaces, estimator, maxDspNum);
  hierFuncSpace.combFuncDesignSpaces();

  return hierFuncSpace;
}

/// DSE Stage3: Explore the function design space through dynamic programming.
FuncDesignSpace ScaleHLSExplorer::exploreDesignSpace(func::FuncOp func, bool directiveOnly,
                                          StringRef outputRootPath,
                                          StringRef csvRootPath) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage3: Conduct top function design "
                             "space exploration...\n";);

  auto tmpFunc = func.clone();
  AffineLoopBands targetBands;
  getLoopBands(tmpFunc.front(), targetBands);
  unsigned targetNum = targetBands.size();

  llvm::errs() << "[DSE] Found " << targetNum << " loop band(s) in function '" 
               << func.getName() << "'\n";

  // Search for the pareto frontiers of each target loop band.
  SmallVector<LoopDesignSpace, 4> loopSpaces;
  for (unsigned i = 0; i < targetNum; ++i) {
    auto space =
        LoopDesignSpace(tmpFunc, targetBands[i], estimator, maxDspNum,
                        maxExplParallel, maxLoopParallel, directiveOnly);

    LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    space.initializeLoopDesignSpace(maxInitParallel);

    LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    space.exploreLoopDesignSpace(maxIterNum, maxDistance);
    loopSpaces.push_back(space);

    // Dump design points to csv file for each loop band.
    auto loopCsvFilePath = csvRootPath.str() + func.getName().str() + "_loop_" +
                           std::to_string(i) + "_space.csv";
    space.dumpLoopDesignSpace(loopCsvFilePath);
  }

  // Combine all loop design spaces into a function design space.
  tmpFunc = func.clone();
  auto funcSpace = FuncDesignSpace(tmpFunc, loopSpaces, estimator, maxDspNum);
  funcSpace.combLoopDesignSpaces();

  // Dump design points to csv file for each function.
  auto funcCsvFilePath =
      csvRootPath.str() + func.getName().str() + "_space.csv";
  funcSpace.dumpFuncDesignSpace(funcCsvFilePath);

  // Export sampled pareto points MLIR source.
  funcSpace.exportParetoDesigns(outputNum, outputRootPath);

  // Apply the best function design point under the constraints.
  for (auto &funcPoint : funcSpace.paretoPoints) {
    if (funcPoint.dspNum <= maxDspNum) {
      std::vector<FactorList> tileLists;
      SmallVector<unsigned, 4> targetIIs;

      for (unsigned i = 0; i < targetNum; ++i) {
        auto &loopSpace = funcSpace.loopDesignSpaces[i];
        auto &loopPoint = funcPoint.loopDesignPoints[i];
        auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
        auto targetII = loopPoint.targetII;

        LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": "
                                << "Loop tiling & pipelining (";);
        LLVM_DEBUG(for (auto tile : tileList) { llvm::dbgs() << tile << ","; });
        LLVM_DEBUG(llvm::dbgs() << targetII << ")\n");

        tileLists.push_back(tileList);
        targetIIs.push_back(targetII);
      }

      if (!applyOptStrategy(func, tileLists, targetIIs))
        assert(false && "Failed to apply optimization strategies to the current function");
      break;
    }
  }

  return funcSpace;
}

//===----------------------------------------------------------------------===//
// DesignSpaceExplore Entry
//===----------------------------------------------------------------------===//

/// This is a temporary approach that does not scale.
void ScaleHLSExplorer::applyDesignSpaceExplore(func::FuncOp func,
                                               bool directiveOnly,
                                               StringRef outputRootPath,
                                               StringRef csvRootPath) {
  emitQoRDebugInfo(func, "Start multiple level DSE.");

  // Simplify loop nests by unrolling.
  if (!simplifyLoopNests(func))
    return;

  // Optimize loop bands by loop perfection, loop order permutation, and loop
  // rectangularization.
  if (!optimizeLoopBands(func, directiveOnly))
    return;

  // Explore the design space through a multiple level approach.
  //if (!exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath))
  //  return;
  exploreHierDesignSpace(func, directiveOnly, outputRootPath, csvRootPath);
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
    for (auto func : module.getOps<func::FuncOp>()) {
      if (hasTopFuncAttr(func))
        explorer.applyDesignSpaceExplore(func, directiveOnly, outputPath,
                                         csvPath);
    }
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createDesignSpaceExplorePass(std::string dseTargetSpec) {
  return std::make_unique<DesignSpaceExplore>(dseTargetSpec);
}
