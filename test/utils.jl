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
