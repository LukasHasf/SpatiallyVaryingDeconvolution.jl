@testset "Test full model" begin
    Ny = 64
    Nx = 64
    nrchannels = 1
    nrPSFs = 3
    batchsize = 1
    psfs = rand(Float32, Ny, Nx, nrPSFs)
    model = SpatiallyVaryingDeconvolution.makemodel(psfs)
    img = rand(Float32, Ny, Nx, nrchannels, batchsize)
    prediction = model(img)
    @test ndims(prediction) == ndims(img)
    @test size(prediction) == size(img)
    @test eltype(prediction) == eltype(img)

    @save "testsave.bson" model
    loaded_model = SpatiallyVaryingDeconvolution.loadmodel("testsave.bson")
    @test loaded_model(img) == prediction

    # The same for the 3D model
    Ny = 64
    Nx = 64
    Nz = 64
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

    @save "testsave.bson" model
    loaded_model = SpatiallyVaryingDeconvolution.loadmodel("testsave.bson")
    @test loaded_model(img) == prediction
    rm("testsave.bson")
end