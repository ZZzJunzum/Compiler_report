module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024xf32>) -> (tensor<1024xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<1.700000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<1024xf32>
    %cst_0 = stablehlo.constant dense<2.000000e-01> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %3 = stablehlo.add %1, %2 : tensor<1024xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %5 = stablehlo.maximum %3, %4 : tensor<1024xf32>
    return %5 : tensor<1024xf32>
  }
}
