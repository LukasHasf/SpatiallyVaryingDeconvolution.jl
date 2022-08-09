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

reshape_size = tuple(size(x)[1:(end-1)]...,1)
slice_dim = ndims(x)
slice_iterator = x -> reshape.(eachslice(x; dims=slice_dim), reshape_size...)
losses = loss.( slice_iterator(x),
                slice_iterator(y))

#=
 Range (min … max):  26.182 ms … 92.337 ms  ┊ GC (min … max): 12.61% … 3.41%
 Time  (median):     40.036 ms              ┊ GC (median):     8.36%
 Time  (mean ± σ):   42.685 ms ± 12.434 ms  ┊ GC (mean ± σ):  10.75% ± 6.58%

   ▃▄       ▂▄ ▁        █                                      
  ▃██▇▇▄▃▆▆▇██▇█▃▇▇▃▄▆▃▆█▇▃▄▄▆▁▄▄▃▁▁▁▁▁▁▁▁▃▁▁▁▃▁▁▁▁▁▃▁▁▁▁▁▁▁▃ ▃
  26.2 ms         Histogram: frequency by time        90.1 ms <

 Memory estimate: 225.94 MiB, allocs estimate: 598.
=#

losses2 = [L1_SSIM_loss(reshape(collect(selectdim(x, ndims(x), i)), size(x)[1:(end-1)]..., 1),
reshape(collect(selectdim(y, ndims(y), i)), size(y)[1:(end-1)]..., 1); kernel=kernel) for i in 1:size(x, ndims(x))]

#=
 Range (min … max):  25.607 ms … 242.354 ms  ┊ GC (min … max): 9.15% … 2.63%
 Time  (median):     46.222 ms               ┊ GC (median):    8.30%
 Time  (mean ± σ):   59.204 ms ±  35.383 ms  ┊ GC (mean ± σ):  9.70% ± 7.57%

  ▂▁  █ █        ▁                                              
  ██▆▆█▅█▆▆▆▅▅▆▃▁█▃▆▁▁▅▃▁▃▃▅▅▆▁▁▁▁▁▅▁▁▃▃▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
  25.6 ms         Histogram: frequency by time          177 ms <

 Memory estimate: 226.17 MiB, allocs estimate: 687.
=#

l1 = L1_SSIM_loss(reshape(x[:, :, 1, 1], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 1], Ny, Nx, nrchannels, 1); kernel=kernel)
l2 = L1_SSIM_loss(reshape(x[:, :, 1, 2], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 2], Ny, Nx, nrchannels, 1); kernel=kernel)
l3 = L1_SSIM_loss(reshape(x[:, :, 1, 3], Ny, Nx, nrchannels, 1),
reshape(y[:, :, 1, 3], Ny, Nx, nrchannels, 1); kernel=kernel)

@test losses == [l1, l2, l3]
@test losses == losses2