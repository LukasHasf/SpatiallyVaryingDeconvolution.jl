module UNet
using Flux

"""    channelsize(x)

Return the size of the channel dimension of `x`.
"""
function channelsize(x)
    return size(x, ndims(x) - 1)
end

function uRelu(x)
    return relu.(x)
end

function uUpsampleNearest(x)
    return upsample_nearest(x, tuple(2 .* ones(Int, ndims(x) - 2)...))
end

struct AttentionBlock
    W_gate::Any
    W_x::Any
    ψ::Any
end

Flux.@functor AttentionBlock

function AttentionBlock(F_g::Integer, F_l::Integer, n_coef::Integer)
    # This skips batchnorms, but batchsize is currently 1 
    W_gate = Conv((1, 1), F_g => n_coef)
    W_x = Conv((1, 1), F_l => n_coef)
    ψ = Conv((1, 1), n_coef => 1, σ)
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
    upsample::Any
    a::Any
end

Flux.@functor UNetUpBlock

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    if u.a != false
        bridge = u.a(x, bridge)
    end
    return cat(x, bridge; dims=ndims(x) - 1)
end

struct ConvBlock
    chain::Chain
    actfun::Any
    residual::Bool
end
Flux.trainable(c::ConvBlock) = (c.chain,)
Flux.@functor ConvBlock

function ConvBlock(
    in_chs::Int,
    out_chs::Int;
    kernel=(3, 3),
    dropout=false,
    activation="relu",
    transpose=false,
    residual=true,
    norm="batch",
)
    if transpose
        conv1 = ConvTranspose(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
        conv2 = ConvTranspose(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)
    else
        conv1 = Conv(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
        conv2 = Conv(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)
    end

    if norm == "batch"
        norm1 = BatchNorm(out_chs)
        norm2 = BatchNorm(out_chs)
    else
        norm1 = identity
        norm2 = identity
    end
    actfun = identity
    if activation == "relu"
        actfun = uRelu
    end

    if dropout
        # Channel-wise droput
        dropout1 = Dropout(0.05; dims=length(kernel) + 1)
        dropout2 = Dropout(0.05; dims=length(kernel) + 1)
    else
        dropout1 = identity
        dropout2 = identity
    end
    chain = Chain(conv1, dropout1, norm1, actfun, conv2, dropout2, norm2)
    return ConvBlock(chain, actfun, residual)
end

function (c::ConvBlock)(x)
    x1 = c.chain(x)
    if c.residual
        selection = 1:min(channelsize(x1), channelsize(x))
        filldimension = [
            size(x)[1:(end - 2)]..., abs(channelsize(x1) - channelsize(x)), size(x)[end]
        ]
        selected_x = selectdim(x, ndims(x) - 1, selection)
        if channelsize(x1) > channelsize(x)
            x1 =
                x1 .+
                cat(selected_x, fill(zero(eltype(x)), filldimension...); dims=ndims(x1) - 1)
        else
            x1 = x1 .+ selected_x
        end
    end
    x1 = c.actfun(x1)
    return x1
end

function ConvDown(chs::Int; kernel=(2, 2), activation=identity)
    return Conv(kernel, chs => chs, activation; stride=2, groups=chs)
end

struct Unet
    conv_down_blocks::Any
    conv_blocks::Any
    up_blocks::Any
    residual::Bool
end

Flux.trainable(u::Unet) = (u.conv_down_blocks, u.conv_blocks, u.up_blocks)

Flux.@functor Unet

function Unet(
    channels::Int=1,
    labels::Int=channels,
    dims=4;
    residual::Bool=false,
    up="nearest",
    down="conv",
    activation="relu",
    norm="batch",
    attention=false,
    depth=4,
    dropout=false,
)
    valid_upsampling_methods = ["nearest", "tconv"]
    @assert up in valid_upsampling_methods "Upsample method \"$up\" not in $(valid_upsampling_methods)."
    kernel_base = tuple(ones(Int, dims - 2)...)
    if down == "conv"
        kernel = kernel_base .* 2
        conv_down_blocks = []
        for i in 1:depth
            c = ConvDown(16 * 2^i; kernel=kernel) # 32, 64, 128, 256, ... input channels
            c.weight .= 0.01 .* c.weight .+ 0.25
            c.bias .*= 0.01
            push!(conv_down_blocks, c)
        end
    end

    conv_kernel = kernel_base .* 3
    conv_blocks = [
        ConvBlock(
            channels,
            32;
            kernel=conv_kernel,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
        ),
    ]
    for i in 1:depth
        second_exponent = i == depth ? i : i + 1
        c = ConvBlock(
            16 * 2^i,
            16 * 2^second_exponent;
            kernel=conv_kernel,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        push!(conv_blocks, c)
    end
    for i in 1:depth
        second_index = i == depth ? labels : 2^(5 + depth - (i + 1))
        c = ConvBlock(
            2^(5 + depth - (i - 1)),
            second_index;
            kernel=conv_kernel,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        push!(conv_blocks, c)
    end

    if residual
        push!(
            conv_blocks,
            ConvBlock(
                channels,
                labels;
                kernel=conv_kernel,
                residual=residual,
                activation=activation,
                norm=norm,
                dropout=dropout,
            ),
        )
    end

    # Only 2D for now
    if attention && dims == 4
        attention_blocks = []
        for i in 1:depth
            nrch = 16 * 2^(depth - (i - 1))
            a = AttentionBlock(nrch, nrch, nrch)
            push!(attention_blocks, a)
        end
    else
        attention_blocks = repeat([false], depth)
    end

    up_blocks = []
    for i in 1:depth
        if up=="nearest"
            upsample_function = uUpsampleNearest
        elseif up=="tconv"
            chs = 16 * 2^(depth - (i - 1))
            upsample_function = ConvTranspose(tuple(2 .* ones(Int, dims - 2)...), chs => chs; stride=2, groups=chs)
        end
        u = UNetUpBlock(upsample_function, attention_blocks[i])
        push!(up_blocks, u)
    end
    return Unet(conv_down_blocks, conv_blocks, up_blocks, residual)
end

function (u::Unet)(x::AbstractArray)
    depth = length(u.conv_down_blocks)
    c0 = x
    c1 = u.conv_blocks[1](c0)
    c2 = u.conv_blocks[2](u.conv_down_blocks[1](c1))
    c3 = u.conv_blocks[3](u.conv_down_blocks[2](c2))
    c4 = u.conv_blocks[4](u.conv_down_blocks[3](c3))
    if depth == 4
        c5 = u.conv_blocks[5](u.conv_down_blocks[4](c4))
    end
    #for i in 1:depth
    #    cs[i + 1] = u.conv_blocks[i + 1](u.conv_down_blocks[i](cs[i]))
    #    println("cs is $(typeof(cs))")
    #end
    if depth == 4
        up1 = u.conv_blocks[6](u.up_blocks[1](c5, c4))
        up2 = u.conv_blocks[7](u.up_blocks[2](up1, c3))
        up3 = u.conv_blocks[8](u.up_blocks[3](up2, c2))
        up4 = u.conv_blocks[9](u.up_blocks[4](up3, c1))
    elseif depth == 3
        up1 = u.conv_blocks[5](u.up_blocks[1](c4, c3))
        up2 = u.conv_blocks[6](u.up_blocks[2](up1, c2))
        up4 = u.conv_blocks[7](u.up_blocks[3](up2, c1))
    end

    #for i in 2:depth
    #    up = u.conv_blocks[depth + i + 1](u.up_blocks[i](up, cs[depth - i + 1]))
    #end
    if u.residual
        up4 = up4 .+ u.conv_blocks[2 * depth + 2](c0)
    end
    return up4
end

end # module
