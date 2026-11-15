module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<16x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg2: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<16x256xf32> {jax.result_info = "result[0]"}, tensor<16x256xf32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %2 = stablehlo.multiply %arg0, %1 : tensor<16x256xf32>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<16x256xf32>
    %5 = stablehlo.add %2, %4 : tensor<16x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x256xf32>
    %7 = stablehlo.maximum %5, %6 : tensor<16x256xf32>
    return %7, %5 : tensor<16x256xf32>, tensor<16x256xf32>
  }
}
