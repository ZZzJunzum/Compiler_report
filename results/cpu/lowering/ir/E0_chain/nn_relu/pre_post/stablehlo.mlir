module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> (tensor<16x256xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}, tensor<i32> {jax.result_info = "result[2]"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %5 = stablehlo.add %2, %4 : tensor<16x256xf32>
    %6 = call @relu(%5) : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %8 = stablehlo.compare  GT, %5, %7,  FLOAT : (tensor<16x256xf32>, tensor<16x256xf32>) -> tensor<16x256xi1>
    %9 = stablehlo.convert %8 : (tensor<16x256xi1>) -> tensor<16x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.reduce(%9 init: %c) applies stablehlo.add across dimensions = [0, 1] : (tensor<16x256xi32>, tensor<i32>) -> tensor<i32>
    %11 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %13 = stablehlo.multiply %6, %12 : tensor<16x256xf32>
    %14 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %16 = stablehlo.add %13, %15 : tensor<16x256xf32>
    %17 = call @relu(%16) : (tensor<16x256xf32>) -> tensor<16x256xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %19 = stablehlo.compare  GT, %17, %18,  FLOAT : (tensor<16x256xf32>, tensor<16x256xf32>) -> tensor<16x256xi1>
    %20 = stablehlo.convert %19 : (tensor<16x256xi1>) -> tensor<16x256xi32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.reduce(%20 init: %c_1) applies stablehlo.add across dimensions = [0, 1] : (tensor<16x256xi32>, tensor<i32>) -> tensor<i32>
    %cst_2 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %23 = stablehlo.multiply %17, %22 : tensor<16x256xf32>
    %cst_3 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %25 = stablehlo.add %23, %24 : tensor<16x256xf32>
    return %25, %10, %21 : tensor<16x256xf32>, tensor<i32>, tensor<i32>
  }
  func.func private @relu(%arg0: tensor<16x256xf32>) -> tensor<16x256xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x256xf32>
    return %1 : tensor<16x256xf32>
  }
}
