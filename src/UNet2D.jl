module UNet2D
using Distributions: Normal
using Flux
include("mybatchnorm.jl")

function BatchNormWrap(x, out_ch)
    x = MyBatchNorm(out_ch)(x)
    return x
end


"""    channelsize(x)

Return the size of the channel dimension of `x`.
"""
function channelsize(x)
  return size(x, ndims(x)-1)
end

function uRelu(x)
  return relu.(x)
end

function uUpsampleNearest(x)
  return upsample_nearest(x, tuple(2 .* ones(Int, ndims(x) - 2)...))
end

function uUpsampleTconv(x)
  chs = channelsize(x)
  return ConvTranspose(tuple(2 .* ones(Int, ndims(x) - 2)...), chs => chs, stride=2, groups=chs)(x)
end

struct AttentionBlock
  W_gate
  W_x
  ψ
end

Flux.@functor AttentionBlock

function AttentionBlock(F_g::Integer, F_l::Integer, n_coef::Integer)
  # This skips batchnorms, but batchsize is currently 1 
  W_gate = Conv((1, 1), F_g => n_coef)
  W_x = Conv((1,1), F_l => n_coef)
  ψ = Conv((1,1), n_coef => 1, σ)
  return AttentionBlock(W_gate, W_x, ψ)
end

function (a::AttentionBlock)(gate, skip)
  g1 = a.W_gate(gate)
  x1 = a.W_x(skip)
  α = a.ψ(relu.(g1 .+ x1))
  out = skip .* α
  return out
end

struct UNetUpBlock
  upsample
  a
end

Flux.@functor UNetUpBlock

function UNetUpBlock(upsample)
  return UNetUpBlock(upsample, false)
end

function (u::UNetUpBlock)(x, bridge)
  x = u.upsample(x)
  if u.a != false
    bridge = u.a(x, bridge)
  end
  return cat(x, bridge, dims = ndims(x) - 1)
end

struct ConvBlock
  chain::Chain
  actfun
  residual::Bool
end
Flux.trainable(c::ConvBlock) = (c.chain,)
Flux.@functor ConvBlock

function ConvBlock(in_chs::Int, out_chs::Int; kernel = (3,3), dropout=false, activation="relu", transpose=false, residual=true, norm="batch")
  if transpose
    conv1 = ConvTranspose(kernel, in_chs => out_chs, pad=1, init=_random_normal)
    conv2 = ConvTranspose(kernel, out_chs => out_chs, pad=1, init=_random_normal)
  else
    conv1 = Conv(kernel, in_chs => out_chs, pad=1)
    conv2 = Conv(kernel, out_chs => out_chs, pad=1)
  end

  if norm=="batch"
    norm1 = x->BatchNormWrap(x, out_chs)
    norm2 = x->BatchNormWrap(x, out_chs)
  else
    norm1 = identity
    norm2 = identity
  end
  actfun = identity
  if activation=="relu"
    actfun = uRelu
  end

  if dropout
    dropout1 = Dropout(0.05, dims=ndims(kernel)+1)
    dropout2 = Dropout(0.05, dims=ndims(kernel)+1)
  else
    dropout1 = identity
    dropout2 = identity
  end
  chain = Chain(conv1, dropout1, norm1, actfun,
  conv2, dropout2, norm2)
  return ConvBlock(chain, actfun, residual)
  
end

function (c::ConvBlock)(x)
    x1 = c.chain(x)
    if c.residual
        selection = 1:min(channelsize(x1), channelsize(x))
        filldimension = [size(x)[1:end-2]..., abs(channelsize(x1) - channelsize(x)), size(x)[end]]
        selected_x = selectdim(x, ndims(x)-1, selection)
        if channelsize(x1) > channelsize(x)
            x1 = x1 .+ cat( selected_x, fill(zero(eltype(x)), filldimension...); dims=ndims(x1)-1)
        else
            x1 = x1 .+ selected_x
        end
    end
    x1 = c.actfun(x1)
    return x1
end

function ConvDown(chs::Int; kernel=(2,2), activation=identity)
  return DepthwiseConv(kernel, chs=>chs, activation, stride=2)
end

struct Unet2D
  conv_down_blocks
  conv_blocks
  up_blocks
  residual::Bool
end

Flux.trainable(u::Unet2D) = (u.conv_down_blocks, u.conv_blocks, u.up_blocks,)

Flux.@functor Unet2D

function Unet2D(channels::Int = 1, labels::Int = channels, dims=4; residual::Bool = false, up="nearest", down="conv", activation=relu, norm="batch", attention=false)
  kernel_base = tuple(ones(Int, dims-2)...)
  if down=="conv"
    kernel = kernel_base .* 2
    c1 = ConvDown(32; kernel=kernel)
    c2 = ConvDown(64; kernel=kernel)
    c3 = ConvDown(128; kernel=kernel)
    c4 = ConvDown(256; kernel=kernel)
    c1.weight .= 0.01 .* c1.weight .+ 0.25
    c2.weight .= 0.01 .* c2.weight .+ 0.25
    c3.weight .= 0.01 .* c3.weight .+ 0.25
    c4.weight .= 0.01 .* c4.weight .+ 0.25
    c1.bias .*= 0.01
    c2.bias .*= 0.01
    c3.bias .*= 0.01
    c4.bias .*= 0.01
    conv_down_blocks = Chain(c1, c2, c3, c4)
  end

  conv_kernel = kernel_base .* 3
  conv_blocks = [ConvBlock(channels, 32; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(32, 64; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(64, 128; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(128, 256; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(256, 256; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(512, 128; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(256, 64; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(128, 32; kernel=conv_kernel, residual=residual, activation=activation, norm=norm),
  ConvBlock(64, labels; kernel=conv_kernel, residual=residual, activation=activation, norm=norm)]

  if residual
    push!(conv_blocks, ConvBlock(channels, labels; kernel=conv_kernel, residual=residual, activation=activation, norm=norm))
  end

  # Only 2D for now
  if attention && dims==4
    attention_blocks = Chain( AttentionBlock(256, 256, 256),
                            AttentionBlock(128, 128, 128), 
                            AttentionBlock(64,64, 64),
                            AttentionBlock(32, 32, 32))
  else
    attention_blocks = [false, false, false, false]
  end


  if up=="nearest"
    upsample = uUpsampleNearest

    up_blocks = Chain(UNetUpBlock(upsample, attention_blocks[1]),
    UNetUpBlock(upsample, attention_blocks[2]),
    UNetUpBlock(upsample, attention_blocks[3]),
    UNetUpBlock(upsample, attention_blocks[4]))
  elseif up=="tconv"
    error("Upscaling method \"tconv\" not implemented yet")
    upsample2 = uUpsampleTconv
    #upsample2(chs) = x -> ConvTranspose((2,2), chs => chs ÷ 2, stride=2, groups=chs)(x)
    
    up_blocks = Chain(UNetUpBlock(upsample2),
    UNetUpBlock(upsample2),
    UNetUpBlock(upsample2),
    UNetUpBlock(upsample2))
  end							  
  Unet2D(conv_down_blocks, conv_blocks, up_blocks, residual)
end

function (u::Unet2D)(x::AbstractArray)
    c0 = x                                           # in_chs channels
    c1 = u.conv_blocks[1](c0)                        # 32 channels
    c2 = u.conv_blocks[2](u.conv_down_blocks[1](c1)) # 64 channels
    c3 = u.conv_blocks[3](u.conv_down_blocks[2](c2)) # 128 channels
    c4 = u.conv_blocks[4](u.conv_down_blocks[3](c3)) # 256 channels
    c5 = u.conv_blocks[5](u.conv_down_blocks[4](c4)) # 256 channels
    up1 = u.conv_blocks[6](u.up_blocks[1](c5, c4))   # 128 channels
    up2 = u.conv_blocks[7](u.up_blocks[2](up1, c3))  # 64 channels 
    up3 = u.conv_blocks[8](u.up_blocks[3](up2, c2))  # 32 channels
    up4 = u.conv_blocks[9](u.up_blocks[4](up3, c1))  # labels channels
    if u.residual
        up4 = up4 .+ u.conv_blocks[10](c0)           # labels channels
    end
    return up4
end

end # module