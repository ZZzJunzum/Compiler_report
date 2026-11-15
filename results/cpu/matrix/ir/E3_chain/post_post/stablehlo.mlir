module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<4096x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg2: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg4: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<4096x256xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}, tensor<i32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4096x256xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<4096x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4096x256xf32>
    %5 = stablehlo.add %2, %4 : tensor<4096x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %7 = stablehlo.maximum %5, %6 : tensor<4096x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %9 = stablehlo.compare  GT, %7, %8,  FLOAT : (tensor<4096x256xf32>, tensor<4096x256xf32>) -> tensor<4096x256xi1>
    %10 = stablehlo.convert %9 : (tensor<4096x256xi1>) -> tensor<4096x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %11 = stablehlo.reduce(%10 init: %c) applies stablehlo.add across dimensions = [0, 1] : (tensor<4096x256xi32>, tensor<i32>) -> tensor<i32>
    %12 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4096x256xf32>
    %14 = stablehlo.multiply %7, %13 : tensor<4096x256xf32>
    %15 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<4096x256xf32>
    %17 = stablehlo.add %14, %16 : tensor<4096x256xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %19 = stablehlo.maximum %17, %18 : tensor<4096x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %21 = stablehlo.compare  GT, %19, %20,  FLOAT : (tensor<4096x256xf32>, tensor<4096x256xf32>) -> tensor<4096x256xi1>
    %22 = stablehlo.convert %21 : (tensor<4096x256xi1>) -> tensor<4096x256xi32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %23 = stablehlo.reduce(%22 init: %c_3) applies stablehlo.add across dimensions = [0, 1] : (tensor<4096x256xi32>, tensor<i32>) -> tensor<i32>
    %cst_4 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %25 = stablehlo.multiply %19, %24 : tensor<4096x256xf32>
    %cst_5 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<4096x256xf32>
    %27 = stablehlo.add %25, %26 : tensor<4096x256xf32>
    return %27, %11, %23 : tensor<4096x256xf32>, tensor<i32>, tensor<i32>
  }
}
