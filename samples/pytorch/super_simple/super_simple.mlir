#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
module attributes {torch.debug_module_name = "lenet"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x1x1xf32> {
    %cst = arith.constant dense<-0.0543794036> : tensor<1xf32>
    %cst_0 = arith.constant dense<[[[[0.48566252, -0.389277816], [0.109658301, -0.429232657]]]]> : tensor<1x1x2x2xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x1x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : tensor<1xf32>) outs(%0 : tensor<1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x1x3x3xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %cst_0 : tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) outs(%1 : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1x3x3xf32>) outs(%0 : tensor<1x1x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ugt, %in, %cst_1 : f32
      %9 = arith.select %8, %in, %cst_1 : f32
      linalg.yield %9 : f32
    } -> tensor<1x1x3x3xf32>
    %4 = tensor.empty() : tensor<1x1x1x1xf32>
    %5 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<1x1x3x3xf32>, tensor<2x2xf32>) outs(%5 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    return %7 : tensor<1x1x1x1xf32>
  }
}

