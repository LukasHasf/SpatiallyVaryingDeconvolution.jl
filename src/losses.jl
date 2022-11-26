export L1_SSIM_loss

function nn_convolve(img::AbstractArray{T,N}; kernel::AbstractArray{T,N}) where {T,N}
    convolved = conv(img, kernel;)
    return convolved
end

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

function L1_loss(ŷ, y)
    return Flux.Losses.mae(y, ŷ)
end

"""    spectral_loss(ŷ, y)

Calculate difference in power spectral density of `ŷ` and `y`.
"""
function spectral_loss(ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    dims = 1:(ndims(y)-2)
    ps_gt = log.(one(T) .+ abs.(fft(y, dims)))
    ps_pred = log.(one(T) .+ abs.(fft(ŷ, dims)))
    ps_gt_norm = ps_gt ./ sum(ps_gt)
    ps_pred_norm = ps_pred ./ sum(ps_pred)
    return sum(abs, ps_gt_norm .- ps_pred_norm)
end

function L1_SSIM_loss(ŷ, y; kernel=nothing)
    return L1_loss(ŷ, y) + SSIM_loss(ŷ, y; kernel=kernel)
end
