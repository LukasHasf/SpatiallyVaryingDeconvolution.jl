@testset "Test full model" begin
    # Make Ny, Nx as small as possible for test -> 2^depth = 2^4 = 16
    Ny = 16
    Nx = 16
    nrchannels = 1
    nrPSFs = 3
    batchsize = 1
    psfs = rand(Float32, Ny, Nx, nrPSFs)
    model_settings = Dict{Symbol, Any}(:attention=>false, :dropout=>false, :depth=>3, :separable=>false, :final_attention=>false, :multiscale=>false, :deconv=>"wiener")
    model = SpatiallyVaryingDeconvolution.make_model(psfs, model_settings; on_gpu=false)
    img = rand(Float32, Ny, Nx, nrchannels, batchsize)
    prediction = model(img)
    # Test that inference is working
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0
    )
    loaded_model = SpatiallyVaryingDeconvolution.load_model(
        testsave_path; load_optimizer=false, on_gpu=false
    )
    @test loaded_model(img) == prediction

    # Check that model is differentiable
    kernel = Float32.(_get_default_kernel(2))
    kernel = reshape(kernel, size(kernel)..., 1, 1)
    loss(x, y) =
        let model = model, kernel = kernel
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
    model = SpatiallyVaryingDeconvolution.make_model(psfs, model_settings; on_gpu=false)
    img = rand(Float32, Ny, Nx, Nz, nrchannels, batchsize)
    prediction = model(img)
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    # Saving / loading with optimizer
    opt = Adam(0.1)
    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0; opt=opt
    )
    loaded_model, opt_loaded = SpatiallyVaryingDeconvolution.load_model(
        testsave_path; load_optimizer=true, on_gpu=false
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
        attention=true,
        dropout=false,
        norm="batch",
        separable=true,
        final_attention=false,
        multiscale=true,
    )
    img = rand(Float32, Ny, Nx, nrch, batchsize)

    prediction = model(img)
    # Test that inference is working
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    # Saving / loading without optimizer
    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0
    )
    loaded_model = SpatiallyVaryingDeconvolution.load_model(
        testsave_path; load_optimizer=false
    )
    @test loaded_model(img) == prediction

    kernel = Float32.(_get_default_kernel(2))
    kernel = reshape(kernel, size(kernel)..., 1, 1)
    loss(x, y) =
        let model = model, kernel = kernel
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

@testset "train_model" begin
    Ny = 16
    Nx = 16
    nrchannels = 1
    nrPSFs = 3
    batchsize = 1
    model_settings = Dict{Symbol, Any}(:attention=>false, :dropout=>false, :depth=>3, :separable=>false, :final_attention=>false, :multiscale=>false, :deconv=>"wiener")
    psfs = my_gpu(rand(Float32, Ny, Nx, nrPSFs))
    model = my_gpu(SpatiallyVaryingDeconvolution.make_model(psfs, model_settings))
    train_x = my_gpu(rand(Float32, Ny, Nx, nrchannels, 50))
    train_y = my_gpu(rand(Float32, Ny, Nx, nrchannels, 50))
    test_x = my_gpu(rand(Float32, Ny, Nx, nrchannels, 50))
    test_y = my_gpu(rand(Float32, Ny, Nx, nrchannels, 50))
    kernel = _get_default_kernel(2; T=Float32)
    kernel = my_gpu(reshape(kernel, size(kernel)..., 1, 1))
    loss_fn = let model = model, kernel = kernel
        function loss_fn(x, y)
            return L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    end
    plotdir = mktempdir()
    chkptdir = mktempdir()
    logfile = joinpath(mktempdir(), "logfile.log")
    settings = SpatiallyVaryingDeconvolution.Settings(Dict(), Dict(),
    Dict(:plot_dir=>plotdir, :plot_interval=>1, :optimizer=>Adam(), :epochs=>1, :logfile=>logfile, :early_stopping=>0, :batchsize=>1),
    Dict(:save_interval=>1, :checkpoint_dir=>chkptdir, :epoch_offset=>0))
    SpatiallyVaryingDeconvolution.train_model(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_fn,
        settings;
        plotloss=true,
    )
    @test isfile(logfile)
    @test length(readdir(chkptdir)) == 2
    @test all([endswith(name, ".bson") for name in readdir(chkptdir)])
    @test !isempty(readdir(plotdir))
end

@testset "Saving / loading RLLayer" begin
    model_settings = Dict{Symbol, Any}(:attention=>false, :dropout=>false, :depth=>3, :separable=>false, :final_attention=>false, :multiscale=>false, :deconv=>"rl")
    Ny = 16
    Nx = 16
    nrch = 1
    nrPSFs = 3
    batchsize = 1
    psfs = my_gpu(rand(Float32, Ny, Nx, nrPSFs))
    model = my_gpu(SpatiallyVaryingDeconvolution.make_model(psfs, model_settings))
    @test model[1] isa SpatiallyVaryingDeconvolution.RLLayer.RL
    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0
    )
    loaded_model = SpatiallyVaryingDeconvolution.load_model(
        testsave_path; load_optimizer=false
    )
    @test loaded_model[1] isa SpatiallyVaryingDeconvolution.RLLayer.RL
    loaded_model = cpu(loaded_model)
    model = cpu(model)
    @test loaded_model[1].PSF == model[1].PSF
    @test loaded_model[1].n_iter == model[1].n_iter
    img = rand(Float32, Ny, Nx, nrch, batchsize)
    @test model(img) == loaded_model(img)
end

@testset "Saving / loading RLLayer_FLFM" begin
    model_settings = Dict{Symbol, Any}(:attention=>false, :dropout=>false, :depth=>3, :separable=>false, :final_attention=>false, :multiscale=>false, :deconv=>"rl_flfm")
    Ny = 16
    Nx = 16
    Nz = 16
    nrch = 1
    nrPSFs = 3
    batchsize = 1
    psfs = my_gpu(rand(Float32, Ny, Nx, Nz, nrPSFs))
    model = my_gpu(SpatiallyVaryingDeconvolution.make_model(psfs, model_settings))
    @test model[1] isa SpatiallyVaryingDeconvolution.RLLayer_FLFM.RL_FLFM
    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0
    )
    loaded_model = SpatiallyVaryingDeconvolution.load_model(
        testsave_path; load_optimizer=false
    )
    @test loaded_model[1] isa SpatiallyVaryingDeconvolution.RLLayer_FLFM.RL_FLFM
    loaded_model = cpu(loaded_model)
    model = cpu(model)
    @test loaded_model[1].PSF == model[1].PSF
    @test loaded_model[1].n_iter == model[1].n_iter
    img = rand(Float32, Ny, Nx, 1, nrch, batchsize)
    @test model(img) == loaded_model(img)
end