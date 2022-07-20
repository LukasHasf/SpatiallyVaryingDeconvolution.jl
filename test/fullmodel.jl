@testset "Test full model" begin
    # Make Ny, Nx as small as possible for test -> 2^depth = 2^4 = 16
    Ny = 16
    Nx = 16
    nrchannels = 1
    nrPSFs = 3
    batchsize = 1
    psfs = rand(Float32, Ny, Nx, nrPSFs)
    model = SpatiallyVaryingDeconvolution.makemodel(psfs)
    img = rand(Float32, Ny, Nx, nrchannels, batchsize)
    prediction = model(img)
    # Test that inference is working
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    testsave_path = SpatiallyVaryingDeconvolution.saveModel(model, mktempdir(), [0.0], 1, 0)
    loaded_model = SpatiallyVaryingDeconvolution.loadmodel(
        testsave_path; load_optimizer=false
    )
    @test loaded_model(img) == prediction

    # Check that model is differentiable
    loss(x, y) =
        let model = model
            kernel = Float32.(_get_default_kernel(2))
            SpatiallyVaryingDeconvolution.L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    img2 = rand(Float32, Ny, Nx, nrchannels, batchsize)
    ps = Flux.params(model)
    gradient_without_error = true
    try
        gs = Flux.gradient(ps) do
            loss(img, img2)
        end
    catch ex
        rethrow(ex)
        gradient_without_error = false
    end
    @test gradient_without_error

    # Test inference and saving/loading for 3D UNet
    Ny = 16
    Nx = 16
    Nz = 16
    nrchannels = 1
    nrPSFs = 3
    batchsize = 1
    psfs = rand(Float32, Ny, Nx, Nz, nrPSFs)
    model = SpatiallyVaryingDeconvolution.makemodel(psfs)
    img = rand(Float32, Ny, Nx, Nz, nrchannels, batchsize)
    prediction = model(img)
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    # Saving / loading with optimizer
    opt = ADAM(0.1)
    testsave_path = SpatiallyVaryingDeconvolution.saveModel(model, mktempdir(), [0.0], 1, 0; opt=opt)
    loaded_model, opt_loaded = SpatiallyVaryingDeconvolution.loadmodel(
        testsave_path; load_optimizer=true
    )
    @test loaded_model(img) == prediction
    @test opt_loaded isa typeof(opt)
    @test getfield(opt_loaded, :eta) == getfield(opt, :eta)
end

@testset "Test other UNet parameters" begin
    Ny = 16
    Nx = 16
    nrch = 1
    batchsize = 2
    model = UNet.Unet(
        nrch,
        1,
        4;
        residual=false,
        up="tconv",
        depth=4,
        dropout=false,
        norm="batch",
        separable=true,
        final_attention=false,
    )
    img = rand(Float32, Ny, Nx, nrch, batchsize)

    prediction = model(img)
    # Test that inference is working
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    # Saving / loading without optimizer
    testsave_path = SpatiallyVaryingDeconvolution.saveModel(model, mktempdir(), [0.0], 1, 0)
    loaded_model = SpatiallyVaryingDeconvolution.loadmodel(
        testsave_path; load_optimizer=false
    )
    @test loaded_model(img) == prediction

    loss(x, y) =
        let model = model
            kernel = Float32.(_get_default_kernel(2))
            SpatiallyVaryingDeconvolution.L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    img2 = rand(Float32, Ny, Nx, nrch, batchsize)
    ps = Flux.params(model)
    gradient_without_error = true
    try
        gs = Flux.gradient(ps) do
            loss(img, img2)
        end
    catch ex
        rethrow(ex)
        gradient_without_error = false
    end
    @test gradient_without_error
end
