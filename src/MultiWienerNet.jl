module MultiWienerNet
using Flux
using FFTW

"""    MultiWiener{T,N}

A Wiener-deconvolution layer with multiple learnable kernels.
"""
struct MultiWiener{T, N} 
    PSF::Array{T,N}
    lambda::Array{T,N}
    planH::FFTW.rFFTWPlan{T}
end

MultiWiener(PSFs) = MultiWiener{eltype(PSFs), ndims(PSFs)}(
    PSFs,
    randn(eltype(PSFs), (ones(Int, ndims(PSFs)-1)..., size(PSFs)[end])),
    plan_rfft(copy(PSFs), 1:(ndims(PSFs)-1), flags=FFTW.MEASURE),
)

"""    (m::MultiWiener)(x)

Apply `MultiWiener` layer to image/volume `x`.
"""
function (m::MultiWiener)(x)
    dims = 1:(ndims(m.PSF)-1)
    H = m.planH * m.PSF
    x̂ = rfft(fftshift(x,dims), dims)
    output = conj.(H) .* x̂  ./ (abs2.(H) .+ m.lambda)
    iffted_output = irfft(output, size(x,1), dims)
    return iffted_output
end
Flux.@functor MultiWiener
Flux.trainable(m::MultiWiener) = (m.PSF, m.lambda,)
end # module
