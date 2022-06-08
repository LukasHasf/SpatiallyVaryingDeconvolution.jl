module UNet3D
using UNet: expand_dims, _random_normal
using Flux
include("mybatchnorm.jl")

function BatchNormWrap(x, out_ch)
    x = MyBatchNorm(out_ch)(x)
    return x
end

function uRelu(x)
  return relu.(x)
end

function uUpsampleNearest(x)
  return upsample_nearest(x, (2,2,2))
end

struct UNetUpBlock
  upsample
end

Flux.@functor UNetUpBlock

function (u::UNetUpBlock)(x, bridge)
  x = u.upsample(x)
  return cat(x, bridge, dims = 4)
end

struct ConvBlock
  chain::Chain
  actfun
  residual::Bool
end
Flux.trainable(c::ConvBlock) = (c.chain,)
Flux.@functor ConvBlock

function ConvBlock(in_chs::Int, out_chs::Int; dropout=false, activation="relu", transpose=false, residual=true, norm="batch")
  if transpose
    conv1 = ConvTranspose((3,3,3),in_chs => out_chs, pad=1, init=_random_normal)
    conv2 = ConvTranspose((3,3,3), out_chs => out_chs, pad=1, init=_random_normal)
  else
    conv1 = Conv((3,3,3), in_chs => out_chs, pad=1)
    conv2 = Conv((3,3,3), out_chs => out_chs, pad=1)
  end

  if norm=="batch" # This block won't work for saving weights
    norm1 = x->BatchNormWrap(x,out_chs)
    norm2 = x->BatchNormWrap(x,out_chs)
  else
    norm1 = identity
    norm2 = identity
  end
  actfun = identity
  if activation=="relu"
    actfun = uRelu
  end

  if dropout
    dropout1 = Dropout(0.05, dims=4)
    dropout2 = Dropout(0.05, dims=4)
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
        # `selection` is 1 up to minimum along `channels` dimension
        selection = 1:min(size(x1)[4], size(x)[4])
        filldimension = [size(x)[1], size(x)[2], size(x)[3],
         abs(size(x1)[4] - size(x)[4]), size(x)[5]]
        if size(x1)[4] > size(x)[4]
            x1 = x1 .+ cat(x[:, :, :, selection, :], fill(zero(eltype(x)), filldimension...); dims=4)
        else
            x1 = x1 .+ x[:, :, :, selection, :]
        end
    end
    x1 = c.actfun(x1)
    return x1
end

function ConvDown(chs::Int; kernel=(2,2,2), activation=identity)
  return DepthwiseConv(kernel, chs=>chs, activation, stride=2)
end

struct Unet3D
  conv_down_blocks
  conv_blocks
  up_blocks
  residual::Bool
end

Flux.trainable(u::Unet3D) = (u.conv_down_blocks, u.conv_blocks, u.up_blocks,)

Flux.@functor Unet3D

function Unet3D(channels::Int = 1, labels::Int = channels; residual::Bool = false, up="nearest", down="conv", activation=relu, norm="batch")
  if down=="conv"
    c1 = ConvDown(32)
    c2 = ConvDown(64)
    c3 = ConvDown(128)
    c4 = ConvDown(256)
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

  conv_blocks = [ConvBlock(channels, 32, residual=residual, activation=activation, norm=norm),
  ConvBlock(32, 64, residual=residual, activation=activation, norm=norm),
  ConvBlock(64, 128, residual=residual, activation=activation, norm=norm),
  ConvBlock(128, 256, residual=residual, activation=activation, norm=norm),
  ConvBlock(256, 256, residual=residual, activation=activation, norm=norm),
  ConvBlock(512, 128, residual=residual, activation=activation, norm=norm),
  ConvBlock(256, 64, residual=residual, activation=activation, norm=norm),
  ConvBlock(128, 32, residual=residual, activation=activation, norm=norm),
  ConvBlock(64, labels, residual=residual, activation=activation, norm=norm)]

  if residual
    push!(conv_blocks, ConvBlock(channels, labels, residual=residual, activation=activation, norm=norm))
  end
  if up=="nearest"
    upsample = uUpsampleNearest

    up_blocks = Chain(UNetUpBlock(upsample),
    UNetUpBlock(upsample),
    UNetUpBlock(upsample),
    UNetUpBlock(upsample))
  elseif up=="tconv"
    upsample2(chs) = x -> ConvTranspose((2,2,2), chs => chs ÷ 2, stride=2, groups=chs)(x)
    
    up_blocks = Chain(UNetUpBlock(upsample2(512)),
    UNetUpBlock(upsample2(128)),
    UNetUpBlock(upsample2(64)),
    UNetUpBlock(upsample2(32)))
  end							  
  Unet3D(conv_down_blocks, conv_blocks, up_blocks, residual)
end

function (u::Unet3D)(x::AbstractArray)
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