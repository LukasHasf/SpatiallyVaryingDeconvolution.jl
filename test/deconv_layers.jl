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

@testset "WienerNet with plan" begin
    Ny = 20
    Nx = 20
    nrPSFs = 2
    batchsize = 1
    nrchannels = 1
    PSFs_cpu = rand(Float32, Ny, Nx, nrPSFs)
    PSFs = my_gpu(PSFs_cpu)
    multiwiener = MultiWienerNet.MultiWienerWithPlan(PSFs)
    img = my_gpu(rand(Float32, Ny, Nx, nrchannels, batchsize))
    pred = multiwiener(img)

    @test ndims(pred) == 4
    @test size(pred) == (Ny, Nx, nrchannels * nrPSFs, batchsize)

    wf1 = wiener_filter(img, my_gpu(PSFs_cpu[:, :, 1]), my_gpu(cpu(multiwiener.lambda)[1]))
    wf2 = wiener_filter(img, my_gpu(PSFs_cpu[:, :, 2]), my_gpu(cpu(multiwiener.lambda)[2]))

    @test cpu(wf1)[:, :, 1, 1] ≈ cpu(pred)[:, :, 1, 1]
    @test cpu(wf2)[:, :, 1, 1] ≈ cpu(pred)[:, :, 2, 1]
end

function rl_deconvolution(img, psf, n_iter)
    psf_flipped = reverse(psf)
    myconv(a, b) = irfft(rfft(a) .* rfft(b), size(a, 1))
    rec = one.(img)
    for _ in 1:n_iter
        rec .*= myconv(img ./ myconv(rec, psf), psf_flipped)
    end
    return ifftshift(rec)
end

@testset "RLLayer" begin
    @testset "lucystep" begin
        # Simplest case: 2D and only one PSF
        a = rand(3, 3)
        psf = rand(3, 3)
        psf_ft = rfft(psf)
        psf_ft_conj = conj.(psf_ft)
        rec = one.(a)
        lucy_one_step = RLLayer.lucystep(rec, psf_ft, psf_ft_conj, 1:2, a)
        @test lucy_one_step == rl_deconvolution(a, psf, 1)
    end
end