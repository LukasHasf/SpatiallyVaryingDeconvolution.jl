module SpatiallyVaryingDeconvolution

export start_training, load_model

using YAML
using Images, Colors
using BSON: @save, @load
using Zygote
using Flux
using CUDA
using Statistics
using Dates
using Plots
using ProgressMeter
using Random
include("utils.jl")
include("UNet.jl")
include("MultiWienerNet.jl")
include("RLLayer.jl")
include("RLLayer_FLFM.jl")
include("losses.jl")
include("plotting.jl")
include("model.jl")
include("main.jl")

Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

show_cuda_capability()
# Set random seed for reproducibility
Random.seed!(1)

export start_training

end # module
