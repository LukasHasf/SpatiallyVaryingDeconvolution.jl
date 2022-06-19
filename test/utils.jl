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

@testset "addnoise" begin
    # Test addnoise for images
    img = ones(Float64, 200, 300, 1, 1)
    nimg = addnoise(img)
    # Since img is just ones, dominating noise is Gaussian with std 1 * (rand(Float64) * 0.02 + 0.005)and mean zero => 0 to 0.025
    # But there is still Poisson noise of 1/sqrt(x) where x is 500 ... 5000
    @test size(img) == size(nimg)
    @test inv(sqrt(5000)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float64, 200, 200, 1, 1) .* 1000
    nimg = addnoise(img)
    @test 1000 / sqrt(5000) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10

    # Test addnoise for volumes
    img = ones(Float32, 100, 200, 300, 1, 1)
    nimg = addnoise(img)
    @test size(img) == size(nimg)
    @test inv(sqrt(5000)) < std(nimg) < 0.025 + inv(sqrt(500))
    @test mean(nimg) ≈ 1.0 atol = 0.01

    img = ones(Float32, 100, 200, 300, 1, 1) .* 1000
    nimg = addnoise(img)
    @test 1000 / sqrt(5000) < std(nimg) < 0.025 + 1000 / sqrt(500)
    @test mean(nimg) ≈ 1000.0 atol = 10
end

@testset "applynoise" begin
    # Testing applynoise for images
    scales = [1.0, 20.0, 300.0, 4000.0]
    img_batch = ones(Float32, 200, 300, 1, 5) .* reshape(scales, 1, 1, 1, :)
    noisy_batch = applynoise(img_batch)
    @test size(img_batch) == size(noisy_batch)
    for i in eachindex(scales)
        s = scales[i]
        nimg = img_batch[:, :, :, i]
        @test mean(nimg) ≈ s atol = s * 0.01
        @test s / sqrt(5000) < std(nimg) < 0.025 + s / sqrt(500)
    end

    # Testing applynoise for volumes
    scales = [1.0, 20.0, 300.0, 4000.0]
    img_batch = ones(Float32, 100, 200, 300, 1, 5) .* reshape(scales, 1, 1, 1, 1, :)
    noisy_batch = applynoise(img_batch)
    @test size(img_batch) == size(noisy_batch)
    for i in eachindex(scales)
        s = scales[i]
        nimg = img_batch[:, :, :, :, i]
        @test mean(nimg) ≈ s atol = s * 0.01
        @test s / sqrt(5000) < std(nimg) < 0.025 + s / sqrt(500)
    end
end
