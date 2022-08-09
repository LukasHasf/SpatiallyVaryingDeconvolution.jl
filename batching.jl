using SpatiallyVaryingDeconvolution
using Test

Ny = 100
Nx = 100
nrchannels = 1

x = rand(Float32, Ny, Nx, nrchannels, 3)
y = rand(Float32, Ny, Nx, nrchannels, 3)

kernel = _get_default_kernel(2; T=Float32)
kernel = reshape(kernel, size(kernel)..., 1, 1)

loss = let kernel = kernel
    function loss_fn(x, y)
        return L1_SSIM_loss(x, y; kernel=kernel)
    end
end


losses = loss.( reshape.(eachslice(x; dims=4), size(x)[1:(end-1)]..., 1),
                reshape.(eachslice(y; dims=4), size(y)[1:(end-1)]..., 1))

l1 = L1_SSIM_loss(reshape(x[:, :, 1, 1], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 1], Ny, Nx, nrchannels, 1); kernel=kernel)
l2 = L1_SSIM_loss(reshape(x[:, :, 1, 2], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 2], Ny, Nx, nrchannels, 1); kernel=kernel)
l3 = L1_SSIM_loss(reshape(x[:, :, 1, 3], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 3], Ny, Nx, nrchannels, 1); kernel=kernel)

@test losses == [l1, l2, l3]