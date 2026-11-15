module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<1x56x56x64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg1: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg2: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg4: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg5: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg6: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg7: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg8: tensor<64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg9: tensor<3x3x64x64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg10: tensor<3x3x64x64xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}) -> (tensor<1x56x56x64xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.convolution(%arg0, %arg9) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %4 = stablehlo.subtract %0, %3 : tensor<1x56x56x64xf32>
    %cst = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %6 = stablehlo.add %2, %5 : tensor<1x1x1x64xf32>
    %7 = stablehlo.rsqrt %6 : tensor<1x1x1x64xf32>
    %8 = stablehlo.reshape %arg6 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x1x1x64xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %11 = stablehlo.multiply %4, %10 : tensor<1x56x56x64xf32>
    %12 = stablehlo.reshape %arg5 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %14 = stablehlo.add %11, %13 : tensor<1x56x56x64xf32>
    %15 = call @relu(%14) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %16 = stablehlo.convolution(%15, %arg10) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<3x3x64x64xf32>) -> tensor<1x56x56x64xf32>
    %17 = stablehlo.broadcast_in_dim %arg3, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %18 = stablehlo.broadcast_in_dim %arg4, dims = [3] : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %19 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %20 = stablehlo.subtract %16, %19 : tensor<1x56x56x64xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x1x1x64xf32>
    %22 = stablehlo.add %18, %21 : tensor<1x1x1x64xf32>
    %23 = stablehlo.rsqrt %22 : tensor<1x1x1x64xf32>
    %24 = stablehlo.reshape %arg8 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<1x1x1x64xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %27 = stablehlo.multiply %20, %26 : tensor<1x56x56x64xf32>
    %28 = stablehlo.reshape %arg7 : (tensor<64xf32>) -> tensor<1x1x1x64xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2, 3] : (tensor<1x1x1x64xf32>) -> tensor<1x56x56x64xf32>
    %30 = stablehlo.add %27, %29 : tensor<1x56x56x64xf32>
    %31 = stablehlo.add %arg0, %30 : tensor<1x56x56x64xf32>
    %32 = call @relu(%31) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %34 = stablehlo.compare  GT, %31, %33,  FLOAT : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xi1>
    %35 = stablehlo.convert %34 : (tensor<1x56x56x64xi1>) -> tensor<1x56x56x64xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.reduce(%35 init: %c) applies stablehlo.add across dimensions = [0, 1, 2, 3] : (tensor<1x56x56x64xi32>, tensor<i32>) -> tensor<i32>
    return %32, %36 : tensor<1x56x56x64xf32>, tensor<i32>
  }
  func.func private @relu(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x56x56x64xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x56x56x64xf32>
    return %1 : tensor<1x56x56x64xf32>
  }
}
