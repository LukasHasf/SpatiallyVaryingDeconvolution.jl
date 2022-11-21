module RLLayer
using Flux
using FFTW

struct RL{T}
    PSF::AbstractArray{T}
    n_iter::Int
end
Flux.@functor RL

Flux.trainable(rl::RL) = (rl.PSF)

function RL(PSFs)
    @assert ndims(PSFs) > 2
    return RL(PSFs ./ sum(PSFs, dims=1:ndims(PSFs)-1), 3)
end

function inspect(a, b)
    if any(isnan.(a))
        println("$b contains NaN")
    else
        println("$b doesn't contain NaN")
    end
    return 
end

myconv(a, b, dims) = irfft(rfft(a, dims) .* b, size(a, 1), dims)

function lucystep(e, psf_ft, psf_ft_conj, dims, x)
    denom = myconv(e, psf_ft_conj, dims)
    fraction = x ./ denom
    return e .* myconv(fraction, psf_ft, dims)
end

function anscombe_transform(x::AbstractArray{T}) where {T}
    return T.(2 .* sqrt.(max.(x .+ 3/8, zero(eltype(x)))))
end

function anscombe_transform_inv(x::AbstractArray{T}) where {T}
    return T.((x ./ 2).^2 .- 3/8)
end

function (rl::RL)(x; n=30)
    x = anscombe_transform(x)
    dims = 1:(ndims(rl.PSF)-1)
    otf = rfft(rl.PSF, dims)
    otf_conj = conj.(otf)
    rec = one.(x)
    for i in 1:n
        rec = lucystep(rec, otf, otf_conj, dims, x)
    end
    return anscombe_transform_inv(ifftshift(rec, dims))
end
end # module