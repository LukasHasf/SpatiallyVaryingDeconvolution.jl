module MultiWienerNet
using Flux
using FFTW

"""    MultiWiener{T,N}

A Wiener-deconvolution layer with multiple learnable kernels.
"""
struct MultiWiener{T,N}
    PSF::AbstractArray{T,N}
    lambda::AbstractArray{T,N}
end

struct MultiWienerWithPlan{T, N}
    PSF::AbstractArray{T,N}
    lambda::AbstractArray{T,N}
    plan::Any
    inv_plan::Any
    plan_x::Any
end

function MultiWiener(PSFs::AbstractArray)
    return MultiWiener{eltype(PSFs),ndims(PSFs)}(
        PSFs, rand(eltype(PSFs), (ones(Int, ndims(PSFs) - 1)..., size(PSFs)[end]))
    )
end

function toMultiWiener(m)
    return MultiWiener(m.PSF, m.lambda)
end

function MultiWienerWithPlan(PSFs)
    m = MultiWiener(PSFs)
    return toMultiWienerWithPlan(m)
end

function toMultiWienerWithPlan(m::MultiWiener)
    nrPSFs = size(m.PSF, ndims(m.PSF))
    plan_x = plan_rfft(similar(m.PSF, size(m.PSF)[1:(ndims(m.PSF)-1)]..., 1, 1), 1:(ndims(m.PSF) - 1))
    plan = plan_rfft(similar(m.PSF), 1:(ndims(m.PSF) - 1))
    dummy_for_inv = similar(m.PSF, complex(eltype(m.PSF)), trunc.(Int, size(m.PSF)[1:(ndims(m.PSF)-1)] .÷ [2, ones(ndims(m.PSF)-2)...] .+ [1, zeros(ndims(m.PSF)-2)...])..., nrPSFs, 1)
    inv_plan = plan_irfft(dummy_for_inv,size(m.PSF, 1), 1:(ndims(dummy_for_inv)-2))
    return MultiWienerWithPlan(m.PSF, m.lambda, plan, inv_plan, plan_x)
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
Flux.trainable(m::MultiWiener) = (m.PSF, m.lambda)

function (m::MultiWienerWithPlan)(x)
    dims = 1:(ndims(m.PSF) - 1)
    H = m.plan * m.PSF
    x̂ = m.plan_x * (fftshift(x, dims))
    #x̂ = rfft(fftshift(x, dims), dims)
    output = conj.(H) .* x̂ ./ (abs2.(H) .+ m.lambda)
    iffted_output = m.inv_plan.scale .* (m.inv_plan.p * output)
    return iffted_output
end
Flux.@functor MultiWienerWithPlan
Flux.trainable(m::MultiWienerWithPlan) = (m.PSF, m.lambda)

end # module
