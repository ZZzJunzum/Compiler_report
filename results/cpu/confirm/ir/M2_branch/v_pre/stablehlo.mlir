module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<64x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg2: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg4: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<64x256xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x256xf32>) -> tensor<64x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<64x256xf32>
    %3 = stablehlo.add %0, %2 : tensor<64x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<64x256xf32>
    %6 = stablehlo.dot_general %5, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x256xf32>, tensor<256x256xf32>) -> tensor<64x256xf32>
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<64x256xf32>
    %9 = stablehlo.add %6, %8 : tensor<64x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %11 = stablehlo.maximum %9, %10 : tensor<64x256xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %12 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %13 = stablehlo.compare  GT, %9, %12,  FLOAT : (tensor<64x256xf32>, tensor<64x256xf32>) -> tensor<64x256xi1>
    %14 = stablehlo.convert %13 : (tensor<64x256xi1>) -> tensor<64x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %15 = stablehlo.reduce(%14 init: %c) applies stablehlo.add across dimensions = [0, 1] : (tensor<64x256xi32>, tensor<i32>) -> tensor<i32>
    %cst_2 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %17 = stablehlo.multiply %11, %16 : tensor<64x256xf32>
    %cst_3 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %19 = stablehlo.add %17, %18 : tensor<64x256xf32>
    %cst_4 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<64x256xf32>
    %21 = stablehlo.multiply %5, %20 : tensor<64x256xf32>
    %22 = stablehlo.add %19, %21 : tensor<64x256xf32>
    return %22, %15 : tensor<64x256xf32>, tensor<i32>
  }
}
