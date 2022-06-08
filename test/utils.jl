@testset "train_test_split" begin
    x = rand(10)
    a, b = train_test_split(x)

    @test length(a) == 7
    @test length(b) == 3
    @test vcat(a, b) == x

    x = rand(10)
    a, b = train_test_split(x, ratio=0.4)
    @test length(a) == 4
    @test length(b) == 6
    @test vcat(a, b) == x

    x = rand(10, 10, 10)
    a, b = train_test_split(x, dim=2)
    @test ndims(a) == 3
    @test ndims(b) == 3
    @test size(a) == (10, 7, 10)
    @test size(b) == (10, 3, 10)
    @test cat(a, b, dims=2) == x
end