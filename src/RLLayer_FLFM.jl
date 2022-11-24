module RLLayer_FLFM
using Flux
using FFTW

struct RL_FLFM{T}
    PSF::AbstractArray{T, 4}
    n_iter::Int
end
Flux.@functor RL_FLFM

Flux.trainable(rl::RL_FLFM) = (rl.PSF)

"""    RL_FLFM(PSFs)

A deconvolution layer using Richardson-Lucy like deconvolution for
the `PSFs` of a Fourier Light Field Microscope.

The RL deconvolution will run for `n_iter` iterations. It is recommended to
leave this number small.
"""
function RL_FLFM(PSFs; n_iter=10)
    @assert ndims(PSFs) == 4 "RL_FLFM deconvolution only works with 3D PSFs. For 2D RL deconvolution, use RL."
    return RL(PSFs ./ sum(PSFs, dims=1:2), n_iter)
end

##################Utility#Functions#########################################
"""    pad_array(arr)

Zero-Pad `arr` along its first two dimension to twice it size.
"""
function pad_array(arr::AbstractArray)
    sizey, sizex = size(arr)[1:2]
    othersizes = size(arr)[3:end]
    pad_left = zeros(eltype(arr), sizey, sizex÷2 + mod(sizex, 2), othersizes...)
    pad_right = zeros(eltype(arr), sizey, sizex÷2, othersizes...)
    pad_top = zeros(eltype(arr), sizey÷2 + mod(sizey, 2), 2*sizex, othersizes...)
    pad_bottom = zeros(eltype(arr), sizey÷2, 2*sizex, othersizes...)
    pad1 = hcat(pad_left, arr, pad_right)
    pad2 = vcat(pad_top, pad1, pad_bottom)
    return pad2
end

"""    lower_index(N)
Give the index of where the original data starts in an array that was
padded to twice its size along a dimension which originally had length `N`.
Utility function for `unpad`.
"""
function lower_index(N)
    return Bool(N % 2) ? (N + 3) ÷ 2 : (N + 2) ÷ 2
end

"""    upper_index(N)
Give the index of where the original data ends in an array that was
padded to twice its size along a dimension which originally had length `N`.
Utility function for `unpad`.
"""
function upper_index(N)
    return Bool(N % 2) ? 3 * N ÷ 2 + 1 : 3 * N ÷ 2
end

"""    conv2_zygote(A, B)

Zero padded convolution of `A` and `B` along the first two dimensions.

`A` and `B` have to have the same size along dimension `1` and `2`.

Differentiable.
"""
function conv2_zygote(A, B)
    sizeA = size(A)[1:2]
    sizeB = size(B)[1:2]
    newsize = max.(sizeA, sizeB)
    A_padded = pad_array(A)
    B_padded = pad_array(B)
    Â = rfft(A_padded, 1:2)
    B̂ = rfft(B_padded, 1:2)
    c = fftshift(irfft(Â .* B̂, 2 * newsize[1], 1:2), 1:2)
    d = selectdim(selectdim(c, 1, lower_index(newsize[1]):upper_index(newsize[1])), 2, lower_index(newsize[2]):upper_index(newsize[2]))
    return d
end

function forward_project(psf, vol)
    return sum(conv2_zygote(psf, vol); dims=3)
end

function backward_project(psf, obs)
    conv2_zygote(psf, obs)
end
############################################################################

function lucystep_flfm(e, psf, psf_flipped, x)
    # https://github.com/ShuJiaLab/HR-FLFM/blob/main/HRFLFM_DataProc/DeconvRL_3D_GPU_HUA.m
    # Maybe also use tanh activation after each step? https://arxiv.org/pdf/2002.01053.pdf
    denom = forward_project(psf, e)
    fraction = x ./ denom
    return e .* backward_project(psf_flipped, fraction) 
end

function (rl::RL_FLFM)(x)
    x = anscombe_transform(x)
    h = rl.PSF
    h_flipped = reverse(h; dims=(1,2))
    rec = backward_project(h, x)
    for _ in 1:rl.n_iter
        rec = lucystep_flfm(rec, h, h_flipped, x)
    end
    return anscombe_transform_inv(rec)
end

end # module RLLayer_FLFM