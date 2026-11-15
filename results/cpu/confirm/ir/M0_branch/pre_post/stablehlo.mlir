module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<4x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg2: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg4: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<4x256xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}, tensor<i32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x256xf32>, tensor<256x256xf32>) -> tensor<4x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<4x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %7 = stablehlo.compare  GT, %3, %6,  FLOAT : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xi1>
    %8 = stablehlo.convert %7 : (tensor<4x256xi1>) -> tensor<4x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.reduce(%8 init: %c) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x256xi32>, tensor<i32>) -> tensor<i32>
    %10 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x256xf32>, tensor<256x256xf32>) -> tensor<4x256xf32>
    %11 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4x256xf32>
    %13 = stablehlo.add %10, %12 : tensor<4x256xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %15 = stablehlo.maximum %13, %14 : tensor<4x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %17 = stablehlo.compare  GT, %15, %16,  FLOAT : (tensor<4x256xf32>, tensor<4x256xf32>) -> tensor<4x256xi1>
    %18 = stablehlo.convert %17 : (tensor<4x256xi1>) -> tensor<4x256xi32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.reduce(%18 init: %c_3) applies stablehlo.add across dimensions = [0, 1] : (tensor<4x256xi32>, tensor<i32>) -> tensor<i32>
    %cst_4 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %21 = stablehlo.multiply %15, %20 : tensor<4x256xf32>
    %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %23 = stablehlo.add %21, %22 : tensor<4x256xf32>
    %cst_6 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<4x256xf32>
    %25 = stablehlo.multiply %5, %24 : tensor<4x256xf32>
    %26 = stablehlo.add %23, %25 : tensor<4x256xf32>
    return %26, %9, %19 : tensor<4x256xf32>, tensor<i32>, tensor<i32>
  }
}
