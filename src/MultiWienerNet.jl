module MultiWienerNet
using Flux
using FFTW
using Tullio

include("utils.jl")

"""    MultiWiener{T,N}

A Wiener-deconvolution layer with multiple learnable kernels.
"""
struct MultiWiener{T,N}
    PSF::AbstractArray{T,N}
    lambda::AbstractArray{T,N}
end

struct MultiWienerWithPlan{T,N}
    PSF::AbstractArray{T,N}
    lambda::AbstractArray{T,N}
    plan::Any
    inv_plan::Any
    plan_x::Any
end

function MultiWiener(PSFs::AbstractArray)
    lambda = similar(PSFs, (ones(Int, ndims(PSFs) - 1)..., size(PSFs)[end]))
    fill!(lambda, rand(Float32))
    return MultiWiener{eltype(PSFs),ndims(PSFs)}(PSFs, lambda)
end

function to_multiwiener(m)
    return MultiWiener(m.PSF, m.lambda)
end

function MultiWienerWithPlan(PSFs)
    m = MultiWiener(PSFs)
    return toMultiWienerWithPlan(m)
end

function toMultiWienerWithPlan(m::MultiWiener)
    sim_psf = my_gpu(similar(m.PSF))
    nd = ndims(sim_psf)
    sz = size(sim_psf)
    nrPSFs = size(m.PSF, nd)
    plan_x = plan_rfft(
        similar(sim_psf, sz[1:(nd - 1)]..., 1, 1), 1:(nd - 1)
    )
    plan = plan_rfft(sim_psf, 1:(nd - 1))
    dummy_for_inv = my_gpu(similar(
        sim_psf,
        complex(eltype(sim_psf)),
        trunc.(
            Int,
            sz[1:(nd - 1)] .÷ [2, ones(nd - 2)...] .+
            [1, zeros(nd - 2)...],
        )...,
        nrPSFs,
        1,
    ))
    inv_plan = plan_irfft(dummy_for_inv, size(m.PSF, 1), 1:(ndims(dummy_for_inv) - 2))
    return MultiWienerWithPlan(my_gpu(m.PSF), my_gpu(m.lambda), plan, inv_plan, plan_x)
end

"""    (m::MultiWiener)(x)

Apply `MultiWiener` layer to image/volume `x`.
"""
function (m::MultiWiener)(x)
    dims = 1:(ndims(m.PSF) - 1)
    H = rfft(m.PSF, dims)
    x̂ = rfft(fftshift(x, dims), dims)
    output = conj.(H) .* x̂ ./ (abs2.(H) .+ m.lambda)
    iffted_output = irfft(output, size(x, 1), dims)
    return iffted_output
end
Flux.@functor MultiWiener

function (m::MultiWienerWithPlan)(x)
    dims = 1:(ndims(m.PSF) - 1)
    H = m.plan * m.PSF
    x̂ = m.plan_x * (fftshift(x, dims))
    output = conj.(H) .* x̂ ./ (abs2.(H) .+ m.lambda)
    iffted_output = m.inv_plan.scale .* (m.inv_plan.p * output)
    return iffted_output
end
Flux.@functor MultiWienerWithPlan
Flux.trainable(m::MultiWienerWithPlan) = (m.PSF, m.lambda)

end # module
