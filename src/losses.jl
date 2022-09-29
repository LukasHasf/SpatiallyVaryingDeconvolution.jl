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

function L1_SSIM_loss(ŷ, y; kernel=nothing)
    return L1_loss(ŷ, y) + SSIM_loss(ŷ, y; kernel=kernel)
end
