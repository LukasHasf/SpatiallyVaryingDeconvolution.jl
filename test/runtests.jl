using SpatiallyVaryingDeconvolution
using Test
using Images
using Tullio
using FFTW
using Flux
using BSON: @save
include("../src/MultiWienerNet.jl")

include("wienernet.jl")
include("fullmodel.jl")
include("utils.jl")
include("losses.jl")