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

function rl_deconvolution(img, psf, n_iter, d)
    psf_flipped = reverse(psf; dims=tuple((1:d)...))
    myconv(a, b) = irfft(rfft(a, 1:d) .* rfft(b, 1:d), size(a, 1), 1:d)
    rec = one.(img)
    for _ in 1:n_iter
        rec .*= myconv(img ./ myconv(rec, psf), psf_flipped)
    end
    return ifftshift(rec, 1:d)
end

@testset "RLLayer" begin
    @testset "lucystep" begin
        # Simplest case: 2D and only one PSF
        a = rand(3, 3)
        psf = rand(3, 3)
        psf_ft = rfft(psf)
        psf_ft_flipped = rfft(reverse(psf))
        rec = one.(a)
        lucy_one_step = RLLayer.lucystep(rec, psf_ft, psf_ft_flipped, 1:2, a)
        @test ifftshift(lucy_one_step) == rl_deconvolution(a, psf, 1, 2)
        # 2D and multiple PSFs
        a = rand(3, 3, 1, 1)
        psf = rand(3, 3, 2)
        psf_ft = rfft(psf, 1:2)
        psf_ft_flipped = rfft(reverse(psf; dims=(1, 2)), 1:2)
        rec = one.(a)
        lucy_one_step = RLLayer.lucystep(rec, psf_ft, psf_ft_flipped, 1:2, a)
        @test size(lucy_one_step) == (3, 3, 2, 1)
        b = rl_deconvolution(a[:, :, 1, 1], psf[:, :, 1], 1, 2)
        c = rl_deconvolution(a[:, :, 1, 1], psf[:, :, 2], 1, 2)
        @test ifftshift(lucy_one_step, 1:2)[:, :, 1] == b
        @test ifftshift(lucy_one_step, 1:2)[:, :, 2] == c
    end
    @testset "Applying RLLayer" begin
        psf = rand(3, 3, 2)
        psf = psf ./ sum(psf; dims=1:2)
        rl = RLLayer.RL(psf; n_iter=30)
        @test Flux.trainable(rl) ≈ (psf)
        a = rand(3, 3, 1, 1)
        â = anscombe_transform(a)
        a_rl = rl(a)
        @test size(a_rl) == (3, 3, 2, 1)
        @test a_rl[:, :, 1, 1] ≈
            anscombe_transform_inv(rl_deconvolution(â[:, :, 1, 1], psf[:, :, 1], 30, 2))
        @test a_rl[:, :, 2, 1] ≈
            anscombe_transform_inv(rl_deconvolution(â[:, :, 1, 1], psf[:, :, 2], 30, 2))
    end
end

@testset "RL_FLFM" begin
    @testset "pad_array" begin
        a = rand(2, 2, 1, 1)
        pad_a1 = RLLayer_FLFM.pad_array(a)
        pad_a2 = select_region(a; new_size=(4, 4, 1, 1))
        @test pad_a1 == pad_a2
        a = rand(3, 3, 1, 1)
        pad_a1 = RLLayer_FLFM.pad_array(a)
        pad_a2 = select_region(a; new_size=(6, 6, 1, 1))
        @test pad_a1 == pad_a2
    end

    @testset "upper/lower index" begin
        @test RLLayer_FLFM.lower_index(10) == 6
        @test RLLayer_FLFM.lower_index(11) == 7
        @test RLLayer_FLFM.upper_index(10) == 15
        @test RLLayer_FLFM.upper_index(11) == 17
    end

    @testset "conv2_zygote" begin
        a = rand(3, 3)
        b = rand(3, 3)
        conv1 = RLLayer_FLFM.conv2_zygote(a, b)
        @test size(conv1) == (3, 3)
        a_pad = select_region(a; new_size=(6, 6))
        b_pad = select_region(b; new_size=(6, 6))
        @test conv1 ≈
            select_region(fftshift(irfft(rfft(a_pad) .* rfft(b_pad), 6)); new_size=(3, 3))

        a = rand(3, 3, 2)
        b = rand(3, 3, 2)
        conv1 = RLLayer_FLFM.conv2_zygote(a, b)
        @test size(conv1) == (3, 3, 2)
        a_pad = select_region(a; new_size=(6, 6, 2))
        b_pad = select_region(b; new_size=(6, 6, 2))
        @test conv1 ≈ select_region(
            fftshift(irfft(rfft(a_pad, 1:2) .* rfft(b_pad, 1:2), 6, 1:2), 1:2);
            new_size=(3, 3, 2),
        )
    end

    @testset "forward_project" begin
        a = rand(3, 3, 3)
        psf = rand(3, 3, 3, 1)
        p1 = RLLayer_FLFM.forward_project(psf, a)
        a_pad = select_region(a; new_size=(6, 6, 3))
        psf_pad = select_region(psf; new_size=(6, 6, 3))
        conv = select_region(
            fftshift(irfft(rfft(a_pad, 1:2) .* rfft(psf_pad, 1:2), 6, 1:2), 1:2);
            new_size=(3, 3, 3),
        )
        p2 = sum(conv; dims=3)
        @test size(p1) == (3, 3, 1, 1)
        @test conv ≈ RLLayer_FLFM.conv2_zygote(psf, a)
        @test p1 ≈ p2

        a = rand(3, 3, 3, 1, 1)
        psf = rand(3, 3, 3, 2)
        p1 = RLLayer_FLFM.forward_project(psf, a)
        a_pad = select_region(a; new_size=(6, 6, 3, 1, 1))
        psf_pad = select_region(psf; new_size=(6, 6, 3, 2))
        conv = select_region(
            fftshift(irfft(rfft(a_pad, 1:2) .* rfft(psf_pad, 1:2), 6, 1:2), 1:2);
            new_size=(3, 3, 3, 2, 1),
        )
        p2 = sum(conv; dims=3)
        @test size(p1) == (3, 3, 1, 2, 1)
        @test p1 ≈ p2
    end

    @testset "backward_project" begin
        a = rand(3, 3, 1)
        psf = rand(3, 3, 3)
        p1 = RLLayer_FLFM.backward_project(psf, a)
        @test p1 ≈ RLLayer_FLFM.conv2_zygote(a, psf)

        a = rand(3, 3, 1, 1, 1)
        psf = rand(3, 3, 3, 2)
        p1 = RLLayer_FLFM.backward_project(psf, a)
        @test size(p1) == (3, 3, 3, 2, 1)
        @test p1 ≈ RLLayer_FLFM.conv2_zygote(a, psf)
    end

    @testset "lucystep_flfm" begin
        psf = rand(3, 3, 3, 2)
        x = rand(3, 3, 1, 1, 1)
        psf_flipped = reverse(psf; dims=(1, 2))
        rec = RLLayer_FLFM.backward_project(psf, x)
        onestep = RLLayer_FLFM.lucystep_flfm(rec, psf, psf_flipped, x)
        denom = RLLayer_FLFM.forward_project(psf, rec)
        fraction = x ./ denom
        @test size(onestep) == (3, 3, 3, 2, 1)
        @test onestep ≈ rec .* RLLayer_FLFM.backward_project(psf_flipped, fraction)
    end

    @testset "Apply RL_FLFM" begin
        # Single iteration
        psfs = rand(3, 3, 3, 2)
        rl_flfm = RLLayer_FLFM.RL_FLFM(psfs; n_iter=1)
        @test rl_flfm.PSF == psfs ./ sum(psfs; dims=1:2)
        @test rl_flfm.n_iter == 1
        @test Flux.trainable(rl_flfm) == rl_flfm.PSF
        x = rand(3, 3, 1, 1, 1)
        x̂ = rl_flfm(x)
        psf = psfs ./ sum(psfs; dims=1:2)
        psf_flipped = reverse(psf; dims=(1, 2))
        x̃ = anscombe_transform(x)
        rec = RLLayer_FLFM.backward_project(psf, x̃)
        x̃2 = RLLayer_FLFM.lucystep_flfm(rec, psf, psf_flipped, x̃)
        x̂2 = anscombe_transform_inv(x̃2)
        @test x̂ ≈ x̂2

        # Multiple iterations
        psfs = rand(3, 3, 3, 2)
        rl_flfm = RLLayer_FLFM.RL_FLFM(psfs; n_iter=2)
        @test rl_flfm.PSF == psfs ./ sum(psfs; dims=1:2)
        @test rl_flfm.n_iter == 2
        @test Flux.trainable(rl_flfm) == rl_flfm.PSF
        x = rand(3, 3, 1, 1, 1)
        x̂ = rl_flfm(x)
        psf = psfs ./ sum(psfs; dims=1:2)
        psf_flipped = reverse(psf; dims=(1, 2))
        x̃ = anscombe_transform(x)
        rec = RLLayer_FLFM.backward_project(psf, x̃)
        rec = RLLayer_FLFM.lucystep_flfm(rec, psf, psf_flipped, x̃)
        x̃2 = RLLayer_FLFM.lucystep_flfm(rec, psf, psf_flipped, x̃)
        x̂2 = anscombe_transform_inv(x̃2)
        @test x̂ ≈ x̂2
    end
end
