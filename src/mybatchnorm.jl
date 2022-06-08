using Flux
using Statistics

mutable struct MyBatchNorm{F,V,N,W}
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W     # moving mean
    σ²::W    # moving var
    ϵ::N
    momentum::N
    affine::Bool
    track_stats::Bool
    active::Union{Bool, Nothing}
    chs::Int # number of channels
  end
  
  function MyBatchNorm(chs::Int, λ=identity;
            initβ=Flux.zeros32, initγ=Flux.ones32,
            affine=true, track_stats=true,
            ϵ=1f-5, momentum=0.1f0)
  
    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? Flux.zeros32(chs) : nothing
    σ² = track_stats ? Flux.ones32(chs) : nothing
  
    return MyBatchNorm(λ, β, γ,
              μ, σ², ϵ, momentum,
              affine, track_stats,
              nothing, chs)
  end
  
  Flux.@functor MyBatchNorm
  trainable(bn::MyBatchNorm) = hasaffine(bn) ? (β = bn.β, γ = bn.γ) : (;)
  
  function (BN::MyBatchNorm)(x)
    @assert size(x, ndims(x)-1) == BN.chs
    N = ndims(x)
    reduce_dims = [1:N-2; N]
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    return _norm_layer_forward(BN, x; reduce_dims, affine_shape)
  end
  
  testmode!(m::MyBatchNorm, mode=true) =
    (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
  
  function Base.show(io::IO, l::MyBatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    hasaffine(l) || print(io,  ", affine=false")
    print(io, ")")
  end

  hasaffine(l::MyBatchNorm) = l.affine
  istraining() = true
  _isactive(m) = isnothing(m.active) ? istraining() : m.active

  function _norm_layer_forward(
    l, x::AbstractArray{T, N}; reduce_dims, affine_shape,
  ) where {T, N}
    if !_isactive(l) && l.track_stats # testmode with tracked stats
      stats_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
      μ = reshape(l.μ, stats_shape)
      σ² = reshape(l.σ², stats_shape)
    else # trainmode or testmode without tracked stats
      μ = mean(x; dims=reduce_dims)
      σ² = mean((x .- μ).^2; dims=reduce_dims)
      if l.track_stats
        _track_stats!(l, x, μ, σ², reduce_dims) # update moving mean/std
      end
    end
  
    o = _norm_layer_forward(x, μ, σ², l.ϵ)
    hasaffine(l) || return l.λ.(o)
  
    γ = reshape(l.γ, affine_shape)
    β = reshape(l.β, affine_shape)
    return l.λ.(γ .* o .+ β)
  end

  @inline _norm_layer_forward(x, μ, σ², ϵ) = (x .- μ) ./ sqrt.(σ² .+ ϵ)

  function _track_stats!(
    bn, x::AbstractArray{T, N}, μ, σ², reduce_dims,
  ) where {T, N}
    V = eltype(bn.σ²)
    mtm = bn.momentum
    res_mtm = one(V) - mtm
    m = prod(size(x, i) for i in reduce_dims)
  
    μnew = vec(N ∈ reduce_dims ? μ : mean(μ, dims=N))
    σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ², dims=N))
  
    bn.μ = res_mtm .* bn.μ .+ mtm .* μnew
    bn.σ² = res_mtm .* bn.σ² .+ mtm .* (m / (m - one(V))) .* σ²new
    return nothing
  end
  