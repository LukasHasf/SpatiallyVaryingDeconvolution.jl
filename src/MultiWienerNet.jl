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

function MultiWiener(PSFs)
    return MultiWiener{eltype(PSFs),ndims(PSFs)}(
        PSFs, randn(eltype(PSFs), (ones(Int, ndims(PSFs) - 1)..., size(PSFs)[end]))
    )
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
end # module
