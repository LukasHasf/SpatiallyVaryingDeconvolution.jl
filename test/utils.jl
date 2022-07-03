@testset "train_test_split" begin
    x = rand(10)
    a, b = train_test_split(x)

    @test length(a) == 7
    @test length(b) == 3
    @test vcat(a, b) == x

    x = rand(10)
    a, b = train_test_split(x; ratio=0.4)
    @test length(a) == 4
    @test length(b) == 6
    @test vcat(a, b) == x

    x = rand(10, 10, 10)
    a, b = train_test_split(x; dim=2)
    @test ndims(a) == 3
    @test ndims(b) == 3
    @test size(a) == (10, 7, 10)
    @test size(b) == (10, 3, 10)
    @test cat(a, b; dims=2) == x
end

@testset "find_complete" begin
    filenames = ["File$i.txt" for i in 1:20]
    dir1 = mktempdir()
    dir2 = mktempdir()
    for filename in filenames
        io = open(joinpath(dir1, filename), "w")
        close(io)
        io = open(joinpath(dir2, filename), "w")
        close(io)
    end
    complete_list = find_complete(10, dir1, dir2)
    @test length(complete_list) == 10
    @test all([complete_list[i] in filenames for i in 1:length(complete_list)])
    io = open(joinpath(dir1, "onlyInDir1.txt"), "w")
    close(io)
    io = open(joinpath(dir1, "onlyInDir2.txt"), "w")
    close(io)
    complete_list = find_complete(21, dir1, dir2)
    @test length(complete_list) == 20
    @test all([complete_list[i] in filenames for i in 1:length(complete_list)])
    @test !("onlyInDir1.txt" in complete_list)
    @test !("onlyInDir2.txt" in complete_list)
end

@testset "channelsize" begin
    img = rand(Float32, 10, 11, 12, 5)
    vol = rand(Float32, 10, 11, 12, 13, 5)
    @test UNet.channelsize(img) == 12
    @test UNet.channelsize(vol) == 13
    img2 = rand(Float64, 5, 6, 7, 4)
    vol2 = rand(Float64, 5, 6, 7, 8, 4)
    @test UNet.channelsize(img2) == 7
    @test UNet.channelsize(vol2) == 8
end

@testset "uRelu" begin
    x = -10:0.1:10
    @test UNet.uRelu(x) == relu.(x)
end

@testset "uUpsampleTconv" begin
    x = rand(Float32, 10, 10, 1, 1)
    x̂ = UNet.uUpsampleTconv(x)
    @test size(x̂) == (20, 20, 1, 1)
end

@testset "train_real_gradient!" begin
    model = Chain(Dense(1, 10), Dense(10, 1))
    X = my_gpu(rand(1, 100))
    Y = my_gpu(rand(1, 100))
    data = Flux.DataLoader((X, Y), batchsize=1)
    ps = Flux.params(model)
    opt = Flux.Optimise.Descent()
    loss_fn(x, y) = Flux.mse(model(x), y)

    model2 = deepcopy(model)
    model = my_gpu(model)
    model2 = my_gpu(model2)
    ps2 = Flux.params(model2)

    Flux.train!(loss_fn, ps, data, opt)
    train_real_gradient!(loss_fn, ps2, data, opt)
    @test Flux.params(model) == Flux.params(model2)
end

@testset "addnoise" begin
    # Test addnoise for images
    img = ones(Float64, 200, 300, 1, 1)
    nimg = addnoise(img)
    # Since img is just ones, dominating noise is Gaussian with std 1 * (rand(Float64) * 0.02 + 0.005)and mean zero => 0 to 0.025
    # But there is still Poisson noise of 1/sqrt(x) where x is 500 ... 5000
    @test size(img) == size(nimg)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float64, 200, 200, 1, 1) .* 1000
    nimg = addnoise(img)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10

    # Test addnoise for volumes
    img = ones(Float32, 100, 200, 300, 1, 1)
    nimg = addnoise(img)
    @test size(img) == size(nimg)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float32, 100, 200, 300, 1, 1) .* 1000
    nimg = addnoise(img)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10
end

@testset "applynoise" begin
    # Testing applynoise for images
    scales = [1.0, 20.0, 300.0, 4000.0]
    img_batch = ones(Float32, 200, 300, 1, length(scales)) .* reshape(scales, 1, 1, 1, :)
    noisy_batch = applynoise(img_batch)
    @test size(img_batch) == size(noisy_batch)
    for i in eachindex(scales)
        s = scales[i]
        nimg = img_batch[:, :, :, i]
        @test mean(nimg) ≈ s atol = s * 0.01
        @test zero(eltype(nimg)) < std(nimg) < 0.025 + s / sqrt(500)
    end

    # Testing applynoise for volumes
    scales = [1.0, 20.0, 300.0, 4000.0]
    img_batch =
        ones(Float32, 100, 200, 300, 1, length(scales)) .* reshape(scales, 1, 1, 1, 1, :)
    noisy_batch = applynoise(img_batch)
    @test size(img_batch) == size(noisy_batch)
    for i in eachindex(scales)
        s = scales[i]
        nimg = img_batch[:, :, :, :, i]
        @test mean(nimg) ≈ s atol = s * 0.01
        @test zero(eltype(nimg)) < std(nimg) < 0.025 + s / sqrt(500)
    end
end

@testset "Test plotting" begin
    # Plot prediction
    # 2D
    prediction = rand(Float32, 20, 20, 1, 10)
    psf = rand(Float32, 20, 20, 6)
    epoch = 5
    epoch_offset = 1
    plotdirectory = mktempdir()
    SpatiallyVaryingDeconvolution.plot_prediction(
        prediction, psf, epoch, epoch_offset, plotdirectory
    )
    produced_files = readdir(plotdirectory)
    @test "Epoch6_predict.png" in produced_files
    @test "LearnedPSF_epoch6.png" in produced_files

    # 3D
    prediction = rand(Float32, 20, 20, 20, 1, 10)
    psf = rand(Float32, 20, 20, 20, 6)
    epoch = 6
    epoch_offset = 2
    plotdirectory = mktempdir()
    SpatiallyVaryingDeconvolution.plot_prediction(
        prediction, psf, epoch, epoch_offset, plotdirectory
    )
    produced_files = readdir(plotdirectory)
    @test "Epoch8_predict.png" in produced_files
    @test "LearnedPSF_epoch8.png" in produced_files

    # plot losses
    train_loss = rand(Float32, 20)
    test_loss = rand(Float32, 20)
    epoch = 10
    plotdirectory = mktempdir()
    SpatiallyVaryingDeconvolution.plot_losses(train_loss, test_loss, epoch, plotdirectory)
    produced_files = readdir(plotdirectory)
    @test "trainlossplot.png" in produced_files
    @test "testlossplot.png" in produced_files
end

@testset "Test load data" begin
    # First, save some temporary pictures
    imgs = rand(32, 32, 6)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")
    save(joinpath(train_dir, "a.png"), imgs[:, :, 1])
    save(joinpath(train_dir, "b.png"), imgs[:, :, 2])
    save(joinpath(train_dir, "exclusive_train.png"), imgs[:, :, 3])
    save(joinpath(test_dir, "a.png"), imgs[:, :, 4])
    save(joinpath(test_dir, "b.png"), imgs[:, :, 5])
    save(joinpath(test_dir, "exclusive_test.png"), imgs[:, :, 6])
    # Load the pictures and compare
    imgs_x, imgs_y = load_data(5, train_dir, test_dir; newsize=(32, 32), T=Float32)
    @test size(imgs_x) == (32, 32, 1, 2)
    @test size(imgs_y) == (32, 32, 1, 2)
    @test imgs_y[:, :, 1, 1] ≈ imgs[:, :, 1] atol = 1e-1
    @test imgs_y[:, :, 1, 2] ≈ imgs[:, :, 2] atol = 1e-1
    @test imgs_x[:, :, 1, 1] ≈ imgs[:, :, 4] atol = 1e-1
    @test imgs_x[:, :, 1, 2] ≈ imgs[:, :, 5] atol = 1e-1

    # Similarly for 3D volumes
    imgs = rand(32, 32, 32, 6)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")

    save(joinpath(train_dir, "a.h5"), Dict("gt" => imgs[:, :, :, 1]))
    save(joinpath(train_dir, "b.h5"), Dict("gt" => imgs[:, :, :, 2]))
    save(joinpath(train_dir, "exclusive_train.h5"), Dict("gt" => imgs[:, :, :, 3]))
    save(joinpath(test_dir, "a.h5"), Dict("sim" => imgs[:, :, :, 4]))
    save(joinpath(test_dir, "b.h5"), Dict("sim" => imgs[:, :, :, 5]))
    save(joinpath(test_dir, "exclusive_test.h5"), Dict("sim" => imgs[:, :, :, 6]))
    # Load the pictures and compare
    imgs_x, imgs_y = load_data(5, train_dir, test_dir; newsize=(32, 32, 32), T=Float32)
    @test size(imgs_x) == (32, 32, 32, 1, 2)
    @test size(imgs_y) == (32, 32, 32, 1, 2)
    @test imgs_y[:, :, :, 1, 1] ≈ imgs[:, :, :, 1]
    @test imgs_y[:, :, :, 1, 2] ≈ imgs[:, :, :, 2]
    @test imgs_x[:, :, :, 1, 1] ≈ imgs[:, :, :, 4]
    @test imgs_x[:, :, :, 1, 2] ≈ imgs[:, :, :, 5]
end

@testset "_help_evaluate_loss" begin
    arr_x = [1 2 3 4]
    arr_y = [5 6 7 8]
    loss_fn(x, y) = x .+ y
    ans = _help_evaluate_loss(arr_x, arr_y, 1, loss_fn)
    @test cpu(ans) == [6;;]
    ans = _help_evaluate_loss(arr_x, arr_y, 4, loss_fn)
    @test cpu(ans) == [12;;]
    ans = _help_evaluate_loss(arr_x, arr_y, 1:3, loss_fn)
    @test cpu(ans) == [6 8 10]
end

@testset "_ensure_existence" begin
    dir = mktempdir()
    mypath = joinpath(dir, "test")
    _ensure_existence(joinpath(dir, "test"))
    _ensure_existence(joinpath(dir, "test2"))
    dirlist = readdir(dir)
    @test length(dirlist) == 2
    @test issetequal(["test", "test2"], dirlist)
end
