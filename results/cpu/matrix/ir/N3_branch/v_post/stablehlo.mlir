module @jit_workload attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<1024x256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg2: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg4: tensor<256xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<1024x256xf32> {jax.result_info = "result[0]"}, tensor<i32> {jax.result_info = "result[1]"}) {
    %0 = chlo.square %arg0 : tensor<1024x256xf32> -> tensor<1024x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %cst_0 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1024x1xf32>
    %4 = stablehlo.divide %2, %3 : tensor<1024x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-5> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1024x1xf32>
    %6 = stablehlo.add %4, %5 : tensor<1024x1xf32>
    %7 = stablehlo.rsqrt %6 : tensor<1024x1xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x256xf32>
    %9 = stablehlo.multiply %arg0, %8 : tensor<1024x256xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<1024x256xf32>
    %13 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
    %15 = stablehlo.add %12, %14 : tensor<1024x256xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %17 = stablehlo.maximum %15, %16 : tensor<1024x256xf32>
    %18 = chlo.square %17 : tensor<1024x256xf32> -> tensor<1024x256xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %19 = stablehlo.reduce(%18 init: %cst_3) applies stablehlo.add across dimensions = [1] : (tensor<1024x256xf32>, tensor<f32>) -> tensor<1024xf32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [0] : (tensor<1024xf32>) -> tensor<1024x1xf32>
    %cst_4 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1024x1xf32>
    %22 = stablehlo.divide %20, %21 : tensor<1024x1xf32>
    %cst_5 = stablehlo.constant dense<9.99999974E-5> : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1024x1xf32>
    %24 = stablehlo.add %22, %23 : tensor<1024x1xf32>
    %25 = stablehlo.rsqrt %24 : tensor<1024x1xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1024x1xf32>) -> tensor<1024x256xf32>
    %27 = stablehlo.multiply %17, %26 : tensor<1024x256xf32>
    %28 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
    %30 = stablehlo.multiply %27, %29 : tensor<1024x256xf32>
    %31 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
    %33 = stablehlo.add %30, %32 : tensor<1024x256xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %35 = stablehlo.maximum %33, %34 : tensor<1024x256xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %37 = stablehlo.compare  GT, %35, %36,  FLOAT : (tensor<1024x256xf32>, tensor<1024x256xf32>) -> tensor<1024x256xi1>
    %38 = stablehlo.convert %37 : (tensor<1024x256xi1>) -> tensor<1024x256xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %39 = stablehlo.reduce(%38 init: %c) applies stablehlo.add across dimensions = [0, 1] : (tensor<1024x256xi32>, tensor<i32>) -> tensor<i32>
    %cst_8 = stablehlo.constant dense<7.500000e-01> : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %41 = stablehlo.multiply %35, %40 : tensor<1024x256xf32>
    %cst_9 = stablehlo.constant dense<1.250000e-01> : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %43 = stablehlo.add %41, %42 : tensor<1024x256xf32>
    %cst_10 = stablehlo.constant dense<2.500000e-01> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1024x256xf32>
    %45 = stablehlo.multiply %17, %44 : tensor<1024x256xf32>
    %46 = stablehlo.add %43, %45 : tensor<1024x256xf32>
    return %46, %39 : tensor<1024x256xf32>, tensor<i32>
  }
}
