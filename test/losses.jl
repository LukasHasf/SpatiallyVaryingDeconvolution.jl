@testset "L1 loss 2D" begin
    N = 6
    img1 = rand(N, N, 1, 1)
    img2 = rand(N, N, 1, 1)
    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img1) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) ==
        SpatiallyVaryingDeconvolution.L1_loss(img2, img1)

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) ≈ sum(abs.(img1 .- img2)) ./ N^2 atol =
        1e-6
end

@testset "L1 loss 3D" begin
    N = 6
    img1 = rand(N, N, N, 1, 1)
    img2 = rand(N, N, N, 1, 1)
    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img1) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) ==
        SpatiallyVaryingDeconvolution.L1_loss(img2, img1)

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) ≈ sum(abs.(img1 .- img2)) ./ N^3 atol =
        1 - 6
end

@testset "SSIM loss 2D" begin
    N = 60
    img1 = rand(Float32, N, N, 1, 1)
    img2 = rand(Float32, N, N, 1, 1)
    kernel = _get_default_kernel(2)
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img1; kernel=kernel) ==
        zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ==
        SpatiallyVaryingDeconvolution.SSIM_loss(img2, img1; kernel=kernel)

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ≈
        one(eltype(img1)) -
          Images.assess_ssim(img1[:, :, 1, 1], img2[:, :, 1, 1]; crop=true) atol = 1e-6
end

@testset "SSIM loss 3D" begin
    N = 60
    img1 = rand(Float32, N, N, N, 1, 1)
    img2 = rand(Float32, N, N, N, 1, 1)
    kernel = _get_default_kernel(3)
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img1; kernel=kernel) ==
        zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ==
        SpatiallyVaryingDeconvolution.SSIM_loss(img2, img1; kernel=kernel)
    # Removing singleton dimension from input greatly speeds up assess_ssim
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ≈
        one(eltype(img1)) -
          Images.assess_ssim(img1[:, :, :, 1, 1], img2[:, :, :, 1, 1]; crop=true) atol =
        1e-6
end

@testset "L1_SSIM loss" begin
    N = 60
    img1 = rand(Float32, N, N, N, 1, 1)
    img2 = rand(Float32, N, N, N, 1, 1)
    kernel = _get_default_kernel(3)
    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img1; kernel=kernel) ==
        zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img2; kernel=kernel) ==
        SpatiallyVaryingDeconvolution.L1_SSIM_loss(img2, img1; kernel=kernel)
    # Removing singleton dimension from input greatly speeds up assess_ssim
    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img2; kernel=kernel) ≈
        sum(abs.(img1 .- img2)) ./ N^3 + one(eltype(img1)) -
          Images.assess_ssim(img1[:, :, :, 1, 1], img2[:, :, :, 1, 1]; crop=true) atol =
        1e-4
end
