@testset "pretty_summarysize" begin
    @test pretty_summarysize(Int32(1)) == "4 bytes"
    @test pretty_summarysize(Int64(1)) == "8 bytes"
    @test pretty_summarysize(Float32(1)) == "4 bytes"
    @test pretty_summarysize(zeros(Float32, 1000)) == "3.945 KiB"
end

@testset "show_cuda_capability" begin
    gpu_available = CUDA.functional() &&
    any([CUDA.capability(dev) for dev in CUDA.devices()] .>= VersionNumber(3, 5, 0))
    teststring = gpu_available ? "Running on GPU" : "Running on CPU"
    @test_logs (:info, teststring) show_cuda_capability()
end

@testset "_center_psfs" begin
    psfs = rand(Float32, 10, 10, 10)
    @test psfs == _center_psfs(psfs, false, -1)
    psfs2, _ = registerPSFs(psfs, psfs[:, :, 6])
    @test psfs2 == _center_psfs(psfs, true, -1)
    psfs3, _ = registerPSFs(psfs, psfs[:, :, 2])
    @test psfs3 == _center_psfs(psfs, true, 2)
end

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

@testset "remove_file_extension" begin
    filename1 = "testimg.img"
    filename2 = "mydir/testimg.img"
    filename3 = "i.have.dots.png"
    @test remove_file_extension(filename1) == "testimg"
    @test remove_file_extension(filename2) == "mydir/testimg"
    @test remove_file_extension(filename3) == "i.have.dots"
end

@testset "find_complete" begin
    # First case of file extensions being the same in both directories
    filenames = ["File$i.txt" for i in 1:20]
    dir1 = mktempdir()
    dir2 = mktempdir()
    for filename in filenames
        io = open(joinpath(dir1, filename), "w")
        close(io)
        io = open(joinpath(dir2, filename), "w")
        close(io)
    end
    complete_list1, complete_list2 = find_complete(10, dir1, dir2)
    # In this case, both lists should be exactly the same
    @test complete_list1 == complete_list2
    # If thats the case, run the tests on one of the lists
    complete_list = complete_list1
    @test length(complete_list) == 10
    @test all([complete_list[i] in filenames for i in eachindex(complete_list)])
    io = open(joinpath(dir1, "onlyInDir1.txt"), "w")
    close(io)
    io = open(joinpath(dir1, "onlyInDir2.txt"), "w")
    close(io)
    complete_list1, complete_list2 = find_complete(21, dir1, dir2)
    @test complete_list1==complete_list2
    complete_list = complete_list1
    @test length(complete_list) == 20
    @test all([complete_list[i] in filenames for i in eachindex(complete_list)])
    @test !("onlyInDir1.txt" in complete_list)
    @test !("onlyInDir2.txt" in complete_list)

    # Second case of file extensions being different in the two directories
    filenames1 = ["File$i.ex1" for i in 1:20]
    filenames2 = ["File$i.ex2" for i in 1:20]
    dir1 = mktempdir()
    dir2 = mktempdir()
    for i in eachindex(filenames1)
        io = open(joinpath(dir1, filenames1[i]), "w")
        close(io)
        io = open(joinpath(dir2, filenames2[i]), "w")
        close(io)
    end
    complete_list1, complete_list2 = find_complete(10, dir1, dir2)
    @test length(complete_list1) == 10 == length(complete_list2)
    @test all([complete_list1[i] in filenames1 for i in eachindex(complete_list1)])
    @test all([complete_list2[i] in filenames2 for i in eachindex(complete_list2)])
    io = open(joinpath(dir1, "onlyInDir1.txt"), "w")
    close(io)
    io = open(joinpath(dir1, "onlyInDir2.txt"), "w")
    close(io)
    complete_list1, complete_list2 = find_complete(21, dir1, dir2)
    @test length(complete_list1) == 20 == length(complete_list2)
    @test !("onlyInDir1.txt" in complete_list1)
    @test !("onlyInDir2.txt" in complete_list1)
    @test !("onlyInDir1.txt" in complete_list2)
    @test !("onlyInDir2.txt" in complete_list2)
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

@testset "anscombe transformations" begin
    a = [1.0 2; 3 4]
    b = 2 .* sqrt.(a .+ 3/8)
    @test anscombe_transform(a) == b
    @test anscombe_transform_inv(anscombe_transform(a)) == a
end

@testset "u_relu" begin
    x = -10:0.1:10
    @test UNet.u_relu(x) == relu.(x)
end

@testset "u_tanh" begin
    x = -10:0.1:10
    @test UNet.u_tanh(x) == tanh.(x)
end

@testset "u_elu" begin
    x = -10:0.1:10
    @test UNet.u_elu(x) == elu.(x)
end

@testset "train_real_gradient!" begin
    # Create two identical models with predetermined weights
    model = Chain(Dense(1, 5), Dense(5, 1))
    model2 = Chain(Dense(1, 5), Dense(5, 1))

    weight1 = Float32.([-0.8949249; 0.46093643; 0.39633024; -0.584888; 0.6077639])
    bias1 = Float32.([0.0, 0.0, 0.0, 0.0, 0.0])
    weight2 = Float32.([-0.389786 0.35682404 -0.40063584 -0.56246924 0.6892315])
    bias2 = Float32.([0.0])
    myparams = [weight1, bias1, weight2, bias2]
    for (i, (p, q)) in enumerate(zip(Flux.params(model), Flux.params(model2)))
        q .= myparams[i]
        p .= myparams[i]
    end

    # Create a predetermined dataset
    X = my_gpu([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    Y = my_gpu([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    data = Flux.DataLoader((X, Y); batchsize=1)
    # Define optimizer and losses; Only need one optimizer since Descent is not stateful
    opt = Flux.Optimise.Descent(0.001)
    loss_fn(x, y) = Flux.mse(model(x), y)
    loss_fn2(x, y) = Flux.mse(model2(x), y)

    # Transfer to gpu if available
    model = my_gpu(model)
    model2 = my_gpu(model2)
    ps = Flux.params(model)
    ps2 = Flux.params(model2)

    # Compare result of Flux.train! with train_real_gradient! in a purely real case -> should be identical
    Flux.train!(loss_fn, ps, data, opt)
    train_real_gradient!(loss_fn2, ps2, data, opt)
    answer = Flux.Params([
        Float32[-0.88388497; 0.45069993; 0.4081554; -0.56870055; 0.5878888],
        Float32[0.002074556, -0.0019173549, 0.0021992736, 0.0030295767, -0.0037178881],
        Float32[-0.36383954 0.34351882 -0.41234216 -0.5456273 0.6717704],
        Float32[-0.0054387837],
    ])
    @test cpu(ps[:]) ≈ answer[:]
    @test cpu(ps[:]) ≈ cpu(ps2[:]) atol=0.01
end

@testset "addnoise" begin
    # Test addnoise for images
    img = ones(Float64, 200, 300, 1, 1)
    nimg = add_noise(img)
    # Since img is just ones, dominating noise is Gaussian with std 1 * (rand(Float64) * 0.02 + 0.005)and mean zero => 0 to 0.025
    # But there is still Poisson noise of 1/sqrt(x) where x is 500 ... 5000
    @test size(img) == size(nimg)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float64, 200, 200, 1, 1) .* 1000
    nimg = add_noise(img)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10

    # Test addnoise for volumes
    img = ones(Float32, 100, 200, 300, 1, 1)
    nimg = add_noise(img)
    @test size(img) == size(nimg)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float32, 100, 200, 300, 1, 1) .* 1000
    nimg = add_noise(img)
    @test zero(eltype(nimg)) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10
end

@testset "applynoise" begin
    # Testing applynoise for images
    scales = [1.0, 20.0, 300.0, 4000.0]
    img_batch = ones(Float32, 200, 300, 1, length(scales)) .* reshape(scales, 1, 1, 1, :)
    noisy_batch = apply_noise(img_batch)
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
    noisy_batch = apply_noise(img_batch)
    @test size(img_batch) == size(noisy_batch)
    for i in eachindex(scales)
        s = scales[i]
        nimg = img_batch[:, :, :, :, i]
        @test mean(nimg) ≈ s atol = s * 0.01
        @test zero(eltype(nimg)) < std(nimg) < 0.025 + s / sqrt(500)
    end
end

@testset "Logging" begin
    epoch = 10
    loss_train = 0.5
    loss_test = 0.56
    logdir = mktempdir()

    # isnothing(logfile) should result in no action
    write_to_logfile(nothing, epoch, loss_train, loss_test)
    @test isempty(readdir(logdir))

    epoch = 11
    loss_train = 0.4
    loss_test = 0.45
    logdir = mktempdir()
    logfile = joinpath(logdir, "losses.log")
    write_to_logfile(logfile, epoch, loss_train, loss_test)
    @test isfile(logfile)
    @test read(logfile, String) == "epoch, train loss, test loss\n11, 0.4, 0.45\n"
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

@testset "_map_to_zero_one" begin
    img = Float32.(rand(Int, 10, 10))
    img_normalized = _map_to_zero_one(img)
    @test size(img) == size(img_normalized)
    @test eltype(img) == eltype(img_normalized)
    min_i, max_i = extrema(img_normalized)
    @test zero(eltype(img_normalized)) <= min_i <= max_i <= one(eltype(img_normalized))

    img = Float32.(rand(Int, 10, 10, 10))
    img_normalized = _map_to_zero_one(img)
    @test size(img) == size(img_normalized)
    @test eltype(img) == eltype(img_normalized)
    min_i, max_i = extrema(img_normalized)
    @test zero(eltype(img_normalized)) <= min_i <= max_i <= one(eltype(img_normalized))
end

@testset "parse_epoch" begin
    teststring = "2022-10-26T20_51_34_loss-0.367_epoch-15.bson"
    @test parse_epoch(teststring) isa Int
    @test parse_epoch(teststring) == 15
    teststring = "some/dirs/before/the/checkpoint/2022-10-26T20_51_34_loss-0.367_epoch-20.bson"
    @test parse_epoch(teststring) isa Int
    @test parse_epoch(teststring) == 20
end

@testset "parse_date" begin
    # parse_date doesn't need to work on paths, only on filenames
    teststring = "2022-10-26T20_51_34_loss-0.367_epoch-15.bson"
    @test parse_date(teststring) isa DateTime
    @test parse_date(teststring)==DateTime(2022, 10, 26, 20, 51, 34)
    teststring = "2021-09-16T22_51_34_loss-0.367_epoch-20.bson"
    @test parse_date(teststring) isa DateTime
    @test parse_date(teststring)==DateTime(2021, 09, 16, 22, 51, 34)
    # Everything that is not a bson file or that is not parseable should return nothing
    teststring = "2021-09-16T22_51_34_loss-0.367_epoch-20.other"
    @test isnothing(parse_date(teststring))
    @test isnothing(parse_date("some_other_filename.bson"))
end

@testset "Test load data" begin
    # First, save some temporary pictures
    imgs = rand(Float32, 32, 32, 6)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")
    dummy_settings = Settings(Dict(:nrsamples=>5, :truth_dir=>train_dir, :sim_dir=>test_dir, :newsize=>(32,32)),
                        Dict(), Dict(), Dict())
    save(joinpath(train_dir, "a.png"), imgs[:, :, 1])
    save(joinpath(train_dir, "b.png"), imgs[:, :, 2])
    save(joinpath(train_dir, "exclusive_train.png"), imgs[:, :, 3])
    save(joinpath(test_dir, "a.png"), imgs[:, :, 4])
    save(joinpath(test_dir, "b.png"), imgs[:, :, 5])
    save(joinpath(test_dir, "exclusive_test.png"), imgs[:, :, 6])
    # Load the pictures and compare
    imgs_x, imgs_y = load_data(dummy_settings; T=Float32)
    @test size(imgs_x) == (32, 32, 1, 2)
    @test size(imgs_y) == (32, 32, 1, 2)
    @test imgs_y[:, :, 1, 1] ≈ imgs[:, :, 1] atol = 1e-1
    @test imgs_y[:, :, 1, 2] ≈ imgs[:, :, 2] atol = 1e-1
    @test imgs_x[:, :, 1, 1] ≈ imgs[:, :, 4] atol = 1e-1
    @test imgs_x[:, :, 1, 2] ≈ imgs[:, :, 5] atol = 1e-1

    # Similarly for 3D volumes
    imgs = rand(Float32, 32, 32, 32, 6)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")
    dummy_settings = Settings(Dict(:nrsamples=>5, :truth_dir=>train_dir, :sim_dir=>test_dir, :newsize=>(32,32,32)),
                        Dict(), Dict(), Dict())

    save(joinpath(train_dir, "a.h5"), Dict("gt" => imgs[:, :, :, 1]))
    save(joinpath(train_dir, "b.h5"), Dict("gt" => imgs[:, :, :, 2]))
    save(joinpath(train_dir, "exclusive_train.h5"), Dict("gt" => imgs[:, :, :, 3]))
    save(joinpath(test_dir, "a.h5"), Dict("sim" => imgs[:, :, :, 4]))
    save(joinpath(test_dir, "b.h5"), Dict("sim" => imgs[:, :, :, 5]))
    save(joinpath(test_dir, "exclusive_test.h5"), Dict("sim" => imgs[:, :, :, 6]))
    # Load the pictures and compare
    imgs_x, imgs_y = load_data(dummy_settings; T=Float32)
    @test size(imgs_x) == (32, 32, 32, 1, 2)
    @test size(imgs_y) == (32, 32, 32, 1, 2)
    @test imgs_y[:, :, :, 1, 1] ≈ imgs[:, :, :, 1]
    @test imgs_y[:, :, :, 1, 2] ≈ imgs[:, :, :, 2]
    @test imgs_x[:, :, :, 1, 1] ≈ imgs[:, :, :, 4]
    @test imgs_x[:, :, :, 1, 2] ≈ imgs[:, :, :, 5]

    # Similarly for 3D volumes in MAT format
    imgs = rand(Float32, 32, 32, 32, 6)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")
    mkdir(train_dir)
    mkdir(test_dir)
    dummy_settings = Settings(Dict(:nrsamples=>5, :truth_dir=>train_dir, :sim_dir=>test_dir, :newsize=>(32,32,32)),
                        Dict(), Dict(), Dict())

    matwrite(joinpath(train_dir, "a.h5"), Dict("gt" => imgs[:, :, :, 1]))
    matwrite(joinpath(train_dir, "b.h5"), Dict("gt" => imgs[:, :, :, 2]))
    matwrite(joinpath(train_dir, "exclusive_train.h5"), Dict("gt" => imgs[:, :, :, 3]))
    matwrite(joinpath(test_dir, "a.h5"), Dict("sim" => imgs[:, :, :, 4]))
    matwrite(joinpath(test_dir, "b.h5"), Dict("sim" => imgs[:, :, :, 5]))
    matwrite(joinpath(test_dir, "exclusive_test.h5"), Dict("sim" => imgs[:, :, :, 6]))
    # Load the pictures and compare
    imgs_x, imgs_y = load_data(dummy_settings; T=Float32)
    @test size(imgs_x) == (32, 32, 32, 1, 2)
    @test size(imgs_y) == (32, 32, 32, 1, 2)
    @test imgs_y[:, :, :, 1, 1] ≈ imgs[:, :, :, 1]
    @test imgs_y[:, :, :, 1, 2] ≈ imgs[:, :, :, 2]
    @test imgs_x[:, :, :, 1, 1] ≈ imgs[:, :, :, 4]
    @test imgs_x[:, :, :, 1, 2] ≈ imgs[:, :, :, 5]
end

@testset "prepare_data" begin
    imgs = rand(Float32, 32, 32, 12)
    img_dir = mktempdir()
    train_dir = joinpath(img_dir, "train")
    test_dir = joinpath(img_dir, "test")
    samples_to_load = 5
    dummy_settings = Settings(Dict(:nrsamples=>samples_to_load, :truth_dir=>train_dir, :sim_dir=>test_dir, :newsize=>(16,17)),
                        Dict(), Dict(), Dict())
    for i in 1:6
        save(joinpath(train_dir, "img_$(i).png"), imgs[:, :, i])
        save(joinpath(test_dir, "img_$(i).png"), imgs[:, :, i])
    end
    train_x, train_y, test_x, test_y = prepare_data(dummy_settings; T=Float32)
    @test all([eltype(x)==Float32 for x in [train_x, train_y, test_x, test_y]])
    split_ind = trunc(Int, 0.7 * samples_to_load)
    @test size(train_x) == (16, 17, 1, split_ind)
    @test size(test_x) == (16, 17, 1, samples_to_load - split_ind)
    @test size(train_x) == size(train_y)
    @test size(test_x) == size(test_y)
    @test all(-1 * one(Float32) .<= train_y .<= one(Float32))
    @test all(-1 * one(Float32) .<= test_y .<= one(Float32))
    # Add a bit of tolerance, since "x" data gets some additional noise after being mapped to range [0, 1]
    @test all(-1.2 * one(Float32) .<= train_x .<= 1.2)
    @test all(-1.2 * one(Float32) .<= test_x .<= 1.2)
end

@testset "prepare_psfs" begin
    psfs_dir = mktempdir()
    psfs = rand(Float64, 16, 16, 4)
    psfs_key = "abc"
    psfs_filename = "a.h5"
    matwrite(joinpath(psfs_dir, psfs_filename), Dict(psfs_key => psfs))
    dummy_settings = SpatiallyVaryingDeconvolution.Settings(Dict(:psfs_path=>joinpath(psfs_dir, psfs_filename), :psfs_key=>psfs_key, :center_psfs=>false, :psf_ref_index=>-1, :newsize=>(16,16)),
                        Dict(), Dict(), Dict())
    psfs_loaded = SpatiallyVaryingDeconvolution.prepare_psfs(dummy_settings; T=Float64)
    @test eltype(psfs_loaded) == Float64
    @test psfs_loaded ≈ psfs
    psfs_loaded = SpatiallyVaryingDeconvolution.prepare_psfs(dummy_settings; T=Float32)
    @test eltype(psfs_loaded) == Float32
    @test psfs_loaded ≈ Float32.(psfs)

    dummy_settings.data[:center_psfs] = true
    psfs_loaded = SpatiallyVaryingDeconvolution.prepare_psfs(dummy_settings)
    @test psfs_loaded ≈ SpatiallyVaryingDeconvolution._center_psfs(psfs, true, -1)

    dummy_settings.data[:center_psfs] = false
    dummy_settings.data[:newsize] = (8,8)
    psfs_loaded = SpatiallyVaryingDeconvolution.prepare_psfs(dummy_settings)
    @test size(psfs_loaded) == (8, 8, 4)
end

@testset "prepare_model!" begin
    psfs_dir = mktempdir()
    psfs = rand(Float64, 16, 16, 4)
    psfs_key = "abc"
    psfs_filename = "a.h5"
    matwrite(joinpath(psfs_dir, psfs_filename), Dict(psfs_key => psfs))
    dummy_settings = Settings(Dict(:psfs_path=>joinpath(psfs_dir, psfs_filename), :psfs_key=>psfs_key, :center_psfs=>false, :psf_ref_index=>-1, :newsize=>(16,16)),
                        Dict(:depth=>3, :attention=>false, :dropout=>true, :separable=>false, :final_attention=>true, :multiscale=>false), 
                        Dict(), 
                        Dict(:load_checkpoints=>false))
    model = prepare_model!(dummy_settings)

    model_true = my_gpu(make_model(my_gpu(prepare_psfs(dummy_settings)), dummy_settings.model))
    @test typeof(model) == typeof(model_true)

    testsave_path = SpatiallyVaryingDeconvolution.save_model(
        model, mktempdir(), [0.0], 1, 0; opt=Flux.AdaGrad()
    )

    dummy_settings = Settings(Dict(:psfs_path=>joinpath(psfs_dir, psfs_filename), :psfs_key=>psfs_key, :center_psfs=>false, :psf_ref_index=>-1, :newsize=>(16,16)),
    Dict(:depth=>3, :attention=>false, :dropout=>true, :separable=>false, :final_attention=>true, :multiscale=>false), 
    Dict(:optimizer=>Flux.Adam()), 
    Dict(:load_checkpoints=>true, :checkpoint_path=>testsave_path))

    loaded_model = prepare_model!(dummy_settings)

    @test typeof(loaded_model) == typeof(model_true)
    @test dummy_settings.training[:optimizer] isa Flux.AdaGrad
end

@testset "_help_evaluate_loss" begin
    arr_x = [1 2 3 4]
    arr_y = [5 6 7 8]
    loss_fn(x, y) = x .+ y
    ans = _help_evaluate_loss(zip(arr_x, arr_y), loss_fn)
    @test cpu(ans) == [6, 8, 10, 12]
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

@testset "Reading options file" begin
    settings = Settings("options.yaml")
    @test settings.data[:sim_dir] == "../../training_data/Data/JuliaForwardModel/"
    @test settings.data[:truth_dir] == "../../training_data/Data/Ground_truth_downsampled/"
    @test settings.data[:newsize] == (64, 64)
    @test settings.data[:center_psfs] == true
    @test settings.data[:psf_ref_index] == -1
    @test settings.model[:depth] == 3
    @test settings.model[:attention] == true
    @test settings.model[:dropout] == true
    @test settings.model[:separable] == true
    @test settings.model[:final_attention] == true
    @test settings.data[:psfs_path] == "../../SpatiallyVaryingConvolution/comaPSF.mat"
    @test settings.data[:psfs_key] == "psfs"
    @test settings.data[:nrsamples] == 700
    @test settings.training[:epochs] == 20
    @test settings.training[:optimizer] isa AdaDelta
    @test settings.training[:plot_interval] == 1
    @test settings.training[:plot_dir] == "examples/training_progress/"
    @test settings.checkpoints[:load_checkpoints] == false
    @test !(:checkpoint_path in keys(settings.checkpoints))
    @test settings.checkpoints[:checkpoint_dir] == "examples/checkpoints/"
    @test settings.checkpoints[:save_interval] == 1
    @test settings.training[:log_losses] == false

    # Latest with no checkpoint found
    settings = Settings("options_latest.yaml")
    @test settings.checkpoints[:load_checkpoints] == false

    # Latest with fake checkpoints
    # Create 2 empty, but correctly named checkpoints, one stray file and an incorrectly named checkpoint file
    path1 = joinpath("examples", "checkpoints", "2022-08-10T14_25_35_loss-0.888_epoch-1.bson")
    path2 = joinpath("examples", "checkpoints", "2022-08-10T15_58_16_loss-0.733_epoch-8.bson")
    path3 = joinpath("examples", "checkpoints", "not_a_bson_file.txt")
    path4 = joinpath("examples", "checkpoints", "date_missing_loss-0.601_epoch-9.bson")
    mkpath(joinpath("examples", "checkpoints"))
    io = open(path1, "w")
    close(io)
    io = open(path2, "w")
    close(io)
    io = open(path3, "w")
    close(io)
    io = open(path4, "w")
    close(io)
    settings =  Settings("options_latest.yaml")
    @test settings.checkpoints[:load_checkpoints] == true
    @test joinpath(split(settings.checkpoints[:checkpoint_path], "/")...) == path2

    # Clean up
    rm(path1)
    rm(path2)
    rm(path3)
    rm(path4)
    rm(joinpath("examples", "checkpoints"))

    settings = Settings("options2.yaml")
    @test settings.checkpoints[:load_checkpoints] == true
    @test joinpath(split(settings.checkpoints[:checkpoint_path],"/")...) == joinpath("examples", "checkpoints", "2022-08-10T15_58_16_loss-0.733_epoch-8.bson")
    @test settings.checkpoints[:epoch_offset] == 8
end

@testset "_get_default_kernel" begin
    kernel2D = _get_default_kernel(2)
    @test size(kernel2D) == (11, 11)
    @test eltype(kernel2D) == Float32
    mykernel = similar(kernel2D)
    mygaussian = gaussian(11, 1.5)
    for j in 1:size(kernel2D, 2)
        for i in 1:size(kernel2D, 1)
            mykernel[i, j] = mygaussian[i] * mygaussian[j]
        end
    end
    @test mykernel == kernel2D

    kernel3D = _get_default_kernel(3)
    @test size(kernel3D) == (11, 11, 11)
    @test eltype(kernel3D) == Float32
    mykernel = similar(kernel3D)
    for k in 1:size(kernel3D, 3)
        for j in 1:size(kernel3D, 2)
            for i in 1:size(kernel3D, 1)
                mykernel[i, j, k] = mygaussian[i] * mygaussian[j] * mygaussian[k]
            end
        end
    end
    @test mykernel == kernel3D
end
