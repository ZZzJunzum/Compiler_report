module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<8x32x32x16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg1: tensor<3x3x16x16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg2: tensor<16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<3x3x16x16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg4: tensor<16xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<8x32x32x16xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x16xf32>, tensor<3x3x16x16xf32>) -> tensor<8x32x32x16xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<8x32x32x16xf32>
    %3 = stablehlo.add %0, %2 : tensor<8x32x32x16xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x32x32x16xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<8x32x32x16xf32>
    %6 = stablehlo.convolution(%5, %arg3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<8x32x32x16xf32>, tensor<3x3x16x16xf32>) -> tensor<8x32x32x16xf32>
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<16xf32>) -> tensor<1x1x1x16xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1, 2, 3] : (tensor<1x1x1x16xf32>) -> tensor<8x32x32x16xf32>
    %9 = stablehlo.add %6, %8 : tensor<8x32x32x16xf32>
    %10 = stablehlo.add %9, %arg0 : tensor<8x32x32x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x32x32x16xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<8x32x32x16xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x32x32x16xf32>
    %14 = stablehlo.compare  GT, %12, %13,  FLOAT : (tensor<8x32x32x16xf32>, tensor<8x32x32x16xf32>) -> tensor<8x32x32x16xi1>
    %15 = stablehlo.convert %14 : (tensor<8x32x32x16xi1>) -> tensor<8x32x32x16xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %16 = stablehlo.reduce(%15 init: %c) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<8x32x32x16xi32>, tensor<i32>) -> tensor<i32>
    %cst_2 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x32x32x16xf32>
    %18 = stablehlo.multiply %12, %17 : tensor<8x32x32x16xf32>
    %cst_3 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x32x32x16xf32>
    %20 = stablehlo.add %18, %19 : tensor<8x32x32x16xf32>
    return %20, %16 : tensor<8x32x32x16xf32>, tensor<i32>
  }
}
