@testset "L1 loss 2D" begin
    N = 6
    img1 = rand(N, N, 1, 1)
    img2 = rand(N, N, 1, 1)
    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img1) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) == SpatiallyVaryingDeconvolution.L1_loss(img2, img1)

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) == sum(abs.(img1 .- img2)) ./ N^2
end

@testset "L1 loss 3D" begin
    N = 6
    img1 = rand(N, N, N, 1, 1)
    img2 = rand(N, N, N, 1, 1)
    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img1) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) == SpatiallyVaryingDeconvolution.L1_loss(img2, img1)

    @test SpatiallyVaryingDeconvolution.L1_loss(img1, img2) == sum(abs.(img1 .- img2)) ./ N^3
end

@testset "SSIM loss 2D" begin
    N = 60
    img1 = rand(N, N, 1, 1)
    img2 = rand(N, N, 1, 1)
    kernel = gaussian(11, 1.5) .* gaussian(11, 1.5)'
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img1; kernel=kernel) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) == SpatiallyVaryingDeconvolution.SSIM_loss(img2, img1; kernel=kernel)

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ≈ one(eltype(img1)) - Images.assess_ssim(img1, img2, crop=true)
end

@testset "SSIM loss 3D" begin
    N = 60
    img1 = rand(N, N, N, 1, 1)
    img2 = rand(N, N, N, 1, 1)
    @tullio kernel[x,y,z] := gaussian(11, 1.5)[x] * gaussian(11, 1.5)[y] *gaussian(11, 1.5)[z]
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img1; kernel=kernel) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) == SpatiallyVaryingDeconvolution.SSIM_loss(img2, img1; kernel=kernel)
    # Removing singleton dimension from input greatly speeds up assess_ssim
    @test SpatiallyVaryingDeconvolution.SSIM_loss(img1, img2; kernel=kernel) ≈ one(eltype(img1)) - Images.assess_ssim(img1[:, :, :, 1, 1], img2[:, :, :, 1, 1], crop=true)
end

@testset "L1_SSIM loss" begin
    N = 60
    img1 = rand(N, N, N, 1, 1)
    img2 = rand(N, N, N, 1, 1)
    @tullio kernel[x,y,z] := gaussian(11, 1.5)[x] * gaussian(11, 1.5)[y] *gaussian(11, 1.5)[z]
    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img1; kernel=kernel) == zero(eltype(img1))

    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img2; kernel=kernel) == SpatiallyVaryingDeconvolution.L1_SSIM_loss(img2, img1; kernel=kernel)
    # Removing singleton dimension from input greatly speeds up assess_ssim
    @test SpatiallyVaryingDeconvolution.L1_SSIM_loss(img1, img2; kernel=kernel) ≈ sum(abs.(img1 .- img2)) ./ N^3 + one(eltype(img1)) - Images.assess_ssim(img1[:, :, :, 1, 1], img2[:, :, :, 1, 1], crop=true)
end