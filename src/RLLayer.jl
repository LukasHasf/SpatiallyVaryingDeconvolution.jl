module RLLayer
using Flux
using FFTW
using Tullio
include("utils.jl")

struct RL{T}
    PSF::AbstractArray{T}
    n_iter::Int
end
Flux.@functor RL

Flux.trainable(rl::RL) = (rl.PSF)

function RL(PSFs, n_iter)
    @assert ndims(PSFs) > 2
    return RL(PSFs ./ sum(PSFs, dims=1:ndims(PSFs)-1), n_iter)
end

myconv(a, b, dims) = irfft(rfft(a, dims) .* b, size(a, 1), dims)

function lucystep(e, psf_ft, psf_ft_conj, dims, x)
    denom = myconv(e, psf_ft, dims)
    fraction = x ./ denom
    return e .* myconv(fraction, psf_ft_conj, dims)
end

function (rl::RL)(x)
    x = anscombe_transform(x)
    dims = 1:(ndims(rl.PSF)-1)
    otf = rfft(rl.PSF, dims)
    otf_conj = rfft(reverse(rl.PSF; dims=tuple(collect(dims)...)), dims)
    rec = one.(x)
    for i in 1:rl.n_iter
        rec = lucystep(rec, otf, otf_conj, dims, x)
    end
    return anscombe_transform_inv(ifftshift(rec, dims))
end
end # module