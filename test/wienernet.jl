function wiener_filter(img, psf, ϵ)
    h = rfft(psf, [1, 2])
    x = rfft(img, [1, 2])
    out = conj.(h) .* x ./ (abs2.(h) .+ ϵ)
    return ifftshift(irfft(out, size(img, 1), [1, 2]))
end

@testset "WienerNet Deconvolution" begin
    Ny = 200
    Nx = 200
    nrPSFs = 2
    batchsize = 1
    nrchannels = 1
    PSFs = rand(Float64, Ny, Nx, nrPSFs)
    multiwiener = MultiWienerNet.MultiWiener(PSFs)
    img = rand(Float64, Ny, Nx, nrchannels, batchsize)
    pred = multiwiener(img)

    @test ndims(pred) == 4
    @test size(pred) == (Ny, Nx, nrchannels * nrPSFs, batchsize)

    wf1 = wiener_filter(img, PSFs[:, :, 1], multiwiener.lambda[1])
    wf2 = wiener_filter(img, PSFs[:, :, 2], multiwiener.lambda[2])

    @test wf1[:, :, 1, 1] ≈ pred[:, :, 1, 1]
    @test wf2[:, :, 1, 1] ≈ pred[:, :, 2, 1]
end
