export L1_SSIM_loss

function nn_convolve(img::AbstractArray{T,N}; kernel::AbstractArray{T,N}) where {T,N}
    convolved = conv(img, kernel;)
    return convolved
end

function rgb_to_lab(a::AbstractArray)
    r = selectdim(a, ndims(a)-1, 1)
    g = selectdim(a, ndims(a)-1, 2)
    b = selectdim(a, ndims(a)-1, 3)
    x = 0.4124564 .* r .+ 0.3575761 .* g .+ 0.1804375 .* b
    y = 0.2126729 .* r .+ 0.7151522 .* g .+ 0.0721750 .* b
    z = 0.0193339 .* r .+ 0.1191920 .* g .+ 0.9503041 .* b
    x_n = 94.811
    y_n = 100.0
    z_n = 107.304
    l = 116 .* (y ./ y_n) .^ (1/3)
    a = 500 .* ((x ./ x_n) .^ (1/3) .- (y ./ y_n) .^ (1/3))
    b = 200 .* ((y ./ y_n) .^ (1/3) .- (z ./ z_n) .^ (1/3))
    # Normalize components
    l_norm = l ./ 25
    a_norm = (a .+ 110) ./ 220
    b_norm = (b .+ 44) ./ 88
    return l_norm, a_norm, b_norm
end

function color_loss(ŷ, y)
    l1, a1, b1 = rgb_to_lab(ŷ)
    l2, a2, b2 = rgb_to_lab(y)
    return L2_loss(l1, l2) + L2_loss(a1, a2) + L2_loss(b1, b2)
end

"""    SSIM_loss(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}; kernel=nothing)

SSIM loss between `ŷ` and `y` using kernel `kernel`.
"""
function SSIM_loss(
    ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}; kernel=nothing
) where {T,N}
    c1 = T(0.01)^2
    c2 = T(0.03)^2
    mu1 = nn_convolve(y; kernel=kernel)
    mu2 = nn_convolve(ŷ; kernel=kernel)
    mu1_sq = mu1 .^ 2
    mu2_sq = mu2 .^ 2
    mu1_mu2 = mu1 .* mu2
    sigma1 = nn_convolve(y .^ 2; kernel=kernel) .- mu1_sq
    sigma2 = nn_convolve(ŷ .^ 2; kernel=kernel) .- mu2_sq
    sigma12 = nn_convolve(y .* ŷ; kernel=kernel) .- mu1_mu2
    ssim_map = @. ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
        ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return one(T) - mean(ssim_map)
end

function SSIM_loss(ŷ::AbstractArray{T,4}, y::AbstractArray{T,5}; kernel=nothing) where {T}
    return SSIM_loss(ŷ, dropdims(y; dims=3); kernel=kernel)
end

"""    L1_loss(ŷ, y)

Mean absolute error between `ŷ` and `y`.
"""
function L1_loss(ŷ, y)
    return Flux.Losses.mae(y, ŷ)
end

function L2_loss(ŷ, y)
    return Flux.Losses.mse(y, ŷ)
end

"""    spectral_loss(ŷ, y)

Calculate difference in power spectral density of `ŷ` and `y`.
"""
function spectral_loss(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    dims = 1:(ndims(y) - 2)
    ps_gt = log.(one(T) .+ abs.(fft(y, dims)))
    ps_pred = log.(one(T) .+ abs.(fft(ŷ, dims)))
    ps_gt_norm = ps_gt ./ sum(ps_gt)
    ps_pred_norm = ps_pred ./ sum(ps_pred)
    return sum(abs, ps_gt_norm .- ps_pred_norm)
end

"""    L1_SSIM_loss(ŷ, y; kernel=nothing)

Sum of the `L1_loss` and the `SSIM_loss`. 
For the calculation of the SSIM-loss, use the given `kernel`.
"""
function L1_SSIM_loss(ŷ, y; kernel=nothing)
    return L1_loss(ŷ, y) + SSIM_loss(ŷ, y; kernel=kernel)
end
