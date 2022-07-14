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

struct SeparableConv
    conv_chain::Chain
end

function SeparableConv(
    filter::NTuple{N,Integer},
    ch::Pair,
    σ=identity;
    stride=1,
    pad=0,
    dilation=1,
    groups=1,
    init=Flux.glorot_uniform,
) where {N}
    convs = []
    for i in 1:N
        filter_dims = Tuple(ones(Int, N))
        filter_ch = i == 1 ? ch : ch[2] => ch[2]
        filter_dims = tuple([n == i ? filter[n] : 1 for n in 1:N]...)
        current_stride = tuple([n == i ? stride : 1 for n in 1:N]...)
        current_pad = tuple([n == i ? pad : 0 for n in 1:N]...)
        current_dilation = tuple([n == i ? dilation : 1 for n in 1:N]...)
        conv = Conv(
            filter_dims,
            filter_ch,
            σ;
            stride=current_stride,
            pad=current_pad,
            dilation=current_dilation,
            groups=groups,
            init=init,
        )
        push!(convs, conv)
    end
    return SeparableConv(Chain(convs...))
end

function (sc::SeparableConv)(x)
    return sc.conv_chain(x)
end

Flux.@functor SeparableConv

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

struct UNetUpBlock{C}
    upsample::Any
    a::Any
    conv_op::C
end

Flux.@functor UNetUpBlock

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    if u.a != false
        bridge = u.a(x, bridge)
    end
    return u.conv_op(cat(x, bridge; dims=ndims(x) - 1))
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
    separable=false
)
    if transpose
        conv1 = ConvTranspose(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
        conv2 = ConvTranspose(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)
    else
        if separable
            conv1 = SeparableConv(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
            conv2 = SeparableConv(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)
        else
            conv1 = Conv(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
            conv2 = Conv(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)
        end
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
    println(size(x))
    println(c.chain[1])
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

function ConvDown(in_chs::Int, out_chs::Int; kernel=(2, 2), conv_kernel=(3,3), activation=identity, residual=false, conv_activation="relu", separable=false, dropout=false, norm="batch")
    downsample_op = Conv(kernel, in_chs => in_chs, activation; stride=2, groups=in_chs)
    conv_op = ConvBlock(in_chs, out_chs; kernel=conv_kernel, dropout=dropout, activation=conv_activation, residual=residual, separable=separable, norm=norm)
    downsample_op.weight .= 0.01 .* downsample_op.weight .+ 0.25
    downsample_op.bias .*= 0.01
    return Chain(downsample_op, conv_op)
end


struct Unet{T,F,R}
    residual_block::R
    residual::Bool
    encoder::T
    decoder::F
end

Flux.trainable(u::Unet) = u.residual ? (u.encoder, u.decoder, u.residual_block) : (u.encoder, u.decoder)

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
    separable=false
)
    valid_upsampling_methods = ["nearest", "tconv"]
    valid_downsampling_methods = ["conv"]
    @assert up in valid_upsampling_methods "Upsample method \"$up\" not in $(valid_upsampling_methods)."
    @assert down in valid_downsampling_methods "Downsampling method \"$down\" not in $(valid_downsampling_methods)."
    kernel_base = tuple(ones(Int, dims - 2)...)
    conv_kernel = kernel_base .* 3
    if down == "conv"
        kernel = kernel_base .* 2
        encoder_blocks = []
        for i in 1:depth
            second_exponent = i == depth ? i : i + 1
            c = ConvDown(16 * 2^i, 16 * 2^second_exponent; kernel=kernel, conv_kernel=conv_kernel, residual=residual, conv_activation=activation, norm=norm, dropout=dropout, separable=separable) # 32, 64, 128, 256, ... input channels
            push!(encoder_blocks, c)
        end
    end

    initial_block = ConvBlock(
            channels,
            32;
            kernel=conv_kernel,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            separable=separable,
        )

    residual_block = nothing
    if residual
        residual_block = ConvBlock(
                channels,
                labels;
                kernel=conv_kernel,
                residual=residual,
                activation=activation,
                norm=norm,
                dropout=dropout,
                separable=separable,
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
        if up == "nearest"
            upsample_function = uUpsampleNearest
        elseif up == "tconv"
            chs = 16 * 2^(depth - (i - 1))
            upsample_function = ConvTranspose(
                tuple(2 .* ones(Int, dims - 2)...), chs => chs; stride=2, groups=chs
            )
        end
        second_index = i == depth ? labels : 2^(5 + depth - (i + 1))
        u = UNetUpBlock(upsample_function, attention_blocks[i], ConvBlock(
            2^(5 + depth - (i - 1)),
            second_index;
            kernel=conv_kernel,
            residual=residual,
            activation=activation,
            norm=norm,
            dropout=dropout,
            separable=separable,
        ))
        push!(up_blocks, u)
    end

    decoder = ntuple(i->up_blocks[i], depth)
    encoder = Chain(initial_block, encoder_blocks...)

    return Unet(residual_block, residual, encoder, decoder)
end

function decode(ops::Tuple, ft::Tuple)
    up = first(ops)(ft[end], ft[end-1])
    #= The next line looks a bit backwards, but the next `up` is calculated
       from the last two elements in ft, not the first =#
    return decode(Base.tail(ops), (ft[1:(end-2)]..., up)) 
end

function decode(::Tuple{}, ft::NTuple{1, T}) where T
    return first(ft)
end

function (u::Unet)(x)
    cs = Flux.activations(u.encoder, x)
    up = decode(u.decoder, cs)
    if u.residual
        up = up .+ u.residual_block(x)
    end
    return up
end

end # module
