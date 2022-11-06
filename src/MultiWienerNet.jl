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

"""    MultiWienerWithPlan(PSFS; on_gpu=true)

Create a `MultiWienerWithPlan` initialized with the PSFs given by `PSFS`.
Uses plan for fourier transforms. 
    
If CUDA capable GPU is available and `on_gpu==true`, uses `rCuFFTPlan`, else use `rFFTWPlan`.
"""
function MultiWienerWithPlan(PSFs; on_gpu=true)
    m = MultiWiener(PSFs)
    return toMultiWienerWithPlan(m; on_gpu=on_gpu)
end

"""    toMultiWienerWithPlan(m; on_gpu=true)

Convert a `MultiWiener` layer that uses `rfft` to a `MultiWienerNet` that uses `rfft plans`.
If `on_gpu`, use `rCuFFTPlan`, else `rFFTWPlan`.
"""
function toMultiWienerWithPlan(m; on_gpu=true)
    to_gpu_cpu = on_gpu ? my_gpu : identity
    sim_psf = to_gpu_cpu(similar(m.PSF))
    nd = ndims(sim_psf)
    sz = size(sim_psf)
    nrPSFs = size(m.PSF, nd)
    plan_x = plan_rfft(similar(sim_psf, sz[1:(nd - 1)]..., 1, 1), 1:(nd - 1))
    plan = plan_rfft(sim_psf, 1:(nd - 1))
    dummy_for_inv = to_gpu_cpu(
        similar(
            sim_psf,
            complex(eltype(sim_psf)),
            trunc.(Int, sz[1:(nd - 1)] .÷ [2, ones(nd - 2)...] .+ [1, zeros(nd - 2)...])...,
            nrPSFs,
            1,
        ),
    )
    inv_plan = plan_irfft(dummy_for_inv, size(m.PSF, 1), 1:(ndims(dummy_for_inv) - 2))
    return MultiWienerWithPlan(to_gpu_cpu(m.PSF), to_gpu_cpu(m.lambda), plan, inv_plan, plan_x)
end

function anscombe_transform(x::AbstractArray{T}) where {T}
    return T.(2 .* sqrt.(max.(x .+ 3/8, zero(eltype(x)))))
end

function anscombe_transform_inv(x::AbstractArray{T}) where {T}
    return T.((x ./ 2).^2 .- 3/8)
end

"""    (m::MultiWiener)(x)

Apply `MultiWiener` layer to image/volume `x`.
"""
function (m::MultiWiener)(x)
    dims = 1:(ndims(m.PSF) - 1)
    H = rfft(m.PSF, dims)
    x̂ = rfft(fftshift(anscombe_transform(x), dims), dims)
    output = conj.(H) .* x̂ ./ (abs2.(H) .+ m.lambda)
    iffted_output = irfft(output, size(x, 1), dims)
    return anscombe_transform_inv(iffted_output)
end
Flux.@functor MultiWiener

function (m::MultiWienerWithPlan)(x)
    dims = 1:(ndims(m.PSF) - 1)
    H = m.plan * m.PSF
    x̂ = m.plan_x * (fftshift(anscombe_transform(x), dims))
    output = conj.(H) .* x̂ ./ (abs2.(H) .+ m.lambda)
    iffted_output = m.inv_plan.scale .* (m.inv_plan.p * output)
    return anscombe_transform_inv(iffted_output)
end
Flux.@functor MultiWienerWithPlan
Flux.trainable(m::MultiWienerWithPlan) = (m.PSF, m.lambda)

end # module
