module UNet
using Flux
using Statistics

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
    W_gate::Conv
    W_x::Conv
    ψ::Conv
end

Flux.@functor AttentionBlock

function AttentionBlock(F_g::Integer, F_l::Integer, n_coef::Integer; dims=4)
    # This skips batchnorms, but batchsize is currently 1 
    kernel = tuple(ones(Int, dims - 2)...)
    W_gate = Conv(kernel, F_g => n_coef)
    W_x = Conv(kernel, F_l => n_coef)
    ψ = Conv(kernel, n_coef => 1, σ)
    return AttentionBlock(W_gate, W_x, ψ)
end

function (a::AttentionBlock)(gate, skip)
    g1 = a.W_gate(gate)
    x1 = a.W_x(skip)
    α = a.ψ(relu.(g1 .+ x1))
    out = skip .* α
    return out
end

function (a::AttentionBlock)(x)
    return a(x, x)
end

struct UNetUpBlock{X, Y, Z}
    upsample::X
    a::Y
    conv_op::Z
end

Flux.@functor UNetUpBlock

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    if u.a != false
        bridge = u.a(x, bridge)
    end
    return u.conv_op(cat(x, bridge; dims=ndims(x) - 1))
end

struct ConvBlock{F}
    chain::Chain
    actfun::F
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
    separable=false,
)
    if transpose
        conv_layer = ConvTranspose
    else
        if separable
            conv_layer = SeparableConv
        else
            conv_layer = Conv
        end
    end

    conv1 = conv_layer(kernel, in_chs => out_chs; pad=1, init=Flux.glorot_normal)
    conv2 = conv_layer(kernel, out_chs => out_chs; pad=1, init=Flux.glorot_normal)

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
    cx1 = channelsize(x1)
    cx = channelsize(x)
    if c.residual
        selection = 1:min(cx1, cx)
        filldimension = [size(x)[1:(end - 2)]..., abs(cx1 - cx), size(x)[end]]
        selected_x = selectdim(x, ndims(x) - 1, selection)
        if cx1 > cx
            x1 = x1 .+
                 cat(selected_x, fill(zero(eltype(x)), filldimension...); dims=ndims(x1) - 1)
        else
            x1 = x1 .+ selected_x
        end
    end
    x1 = c.actfun(x1)
    return x1
end

function ConvDown(
    in_chs::Int,
    out_chs::Int;
    down_kernel=(2, 2),
    kernel=(3, 3),
    down_activation=identity,
    residual=false,
    activation="relu",
    separable=false,
    dropout=false,
    norm="batch",
)
    downsample_op = Conv(
        down_kernel, in_chs => in_chs, down_activation; stride=2, groups=in_chs
    )
    conv_op = ConvBlock(
        in_chs,
        out_chs;
        kernel=kernel,
        dropout=dropout,
        activation=activation,
        residual=residual,
        separable=separable,
        norm=norm,
    )
    downsample_op.weight .= 0.01 .* downsample_op.weight .+ 0.25
    downsample_op.bias .*= 0.01
    return Chain(downsample_op, conv_op)
end

struct SpatialAttentionModule{A}
    conv1::A
end

function SpatialAttentionModule(; dims=4)
    kernel = tuple(ones(Int, dims - 2)...)
    conv1 = Conv(kernel, 2=>1, σ; pad=SamePad())
    return SpatialAttentionModule(conv1)
end

function (sam::SpatialAttentionModule)(x)
    x1 = cat(maximum(x, dims=ndims(x)-1), mean(x, dims=ndims(x)-1); dims=ndims(x)-1)
    return sam.conv1(x1)
end

Flux.@functor SpatialAttentionModule

struct Unet{T,F,R,X,Y}
    residual_block::R
    encoder::T
    decoder::F
    attention_module::X
    upsampler::Y
end

function Flux.trainable(u::Unet)
    return !isnothing(u.residual_block) ? (u.encoder, u.decoder, u.residual_block) : (u.encoder, u.decoder)
end

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
    separable=false,
)
    valid_upsampling_methods = ["nearest", "tconv"]
    valid_downsampling_methods = ["conv"]
    @assert up in valid_upsampling_methods "Upsample method \"$up\" not in $(valid_upsampling_methods)."
    @assert down in valid_downsampling_methods "Downsampling method \"$down\" not in $(valid_downsampling_methods)."
    kernel_base = tuple(ones(Int, dims - 2)...)
    conv_kernel = kernel_base .* 3
    conv_config = (
        residual=residual,
        norm=norm,
        dropout=dropout,
        separable=separable,
        kernel=conv_kernel,
        activation=activation,
    )
    if down == "conv"
        kernel = kernel_base .* 2
        encoder_blocks = []
        for i in 1:depth
            second_exponent = i == depth ? i : i + 1
            c = ConvDown(
                16 * 2^i, 16 * 2^second_exponent; down_kernel=kernel, conv_config...
            ) # 32, 64, 128, 256, ... input channels
            push!(encoder_blocks, c)
        end
    end

    initial_block = ConvBlock(channels, 32; conv_config...)

    residual_block = nothing
    if residual
        residual_block = ConvBlock(channels, labels; conv_config...)
    end

    if attention
        attention_blocks = []
        for i in 1:depth
            nrch = 16 * 2^(depth - (i - 1))
            a = AttentionBlock(nrch, nrch, nrch; dims=dims)
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
        #second_index = i == depth ? labels : 2^(5 + depth - (i + 1))
        second_index = 2^(5 + depth - (i + 1))
        u = UNetUpBlock(
            upsample_function,
            attention_blocks[i],
            ConvBlock(2^(5 + depth - (i - 1)), second_index; conv_config...),
        )
        push!(up_blocks, u)
    end

    decoder = ntuple(i -> up_blocks[i], depth)
    encoder = Chain(initial_block, encoder_blocks...)
    in_channels = sum([2^(3+i) for i in 1:(depth+1)])
    attention_module = Chain(AttentionBlock(in_channels, in_channels, in_channels), Conv((1,1), in_channels=>1; pad=SamePad()))
    #attention_module = Chain(AttentionBlock(16, 16, 16), Conv((1,1), 16=>1; pad=SamePad()))
    upsampler = dims==4 ? upsample_bilinear : upsample_trilinear

    return Unet(residual_block, encoder, decoder, attention_module, upsampler)
end

function decode(ops::Tuple, ft::Tuple)
    up = first(ops)(ft[end], ft[end - 1])
    #= The next line looks a bit backwards, but the next `up` is calculated
       from the last two elements in ft, not the first =#
    return decode(Base.tail(ops), (ft[1:(end - 2)]..., up))..., up
end

function decode(::Tuple{}, ft::NTuple{1,T}) where {T}
    return tuple(first(ft))
end

function (u::Unet)(x)
    cs = Flux.activations(u.encoder, x)
    ups = decode(u.decoder, cs)
    up = first(ups)
    ups = (Base.tail(ups)..., cs[end])
    if !isnothing(u.residual_block)
        up = up .+ u.residual_block(x)
    end
    final_block = ups[1]
    final_size = size(final_block)[1:(end-2)]
    for activation in ups[2:end]
        final_block = cat(final_block, u.upsampler(activation; size=final_size); dims=ndims(final_block)-1)
    end
    #return u.attention_module(up)
    return u.attention_module(final_block)
end

end # module
