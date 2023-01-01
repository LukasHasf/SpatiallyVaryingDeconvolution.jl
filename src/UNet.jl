module UNet
using Flux
using Statistics

const valid_upsampling_methods = ["nearest", "tconv"]
const valid_downsampling_methods = ["conv", "maxpool"]

"""    channelsize(x)

Return the size of the channel dimension of `x`.

# Examples
```julia-repl
julia> channelsize(ones(10, 20, 30, 40))
30
```
"""
function channelsize(x)
    return size(x, ndims(x) - 1)
end

function u_relu(x)
    return relu.(x)
end

function u_tanh(x)
    return tanh.(x)
end

function u_elu(x)
    return elu.(x)
end

function u_upsample_nearest(x)
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
    proper_pad = Flux.calc_padding(Conv, pad, filter, dilation, groups)
    for i in 1:N
        filter_dims = Tuple(ones(Int, N))
        filter_ch = i == 1 ? ch : ch[2] => ch[2]
        filter_dims = tuple([n == i ? filter[n] : 1 for n in 1:N]...)
        current_stride = tuple([n == i ? stride : 1 for n in 1:N]...)
        current_pad = tuple([n == i ? proper_pad[n] : 0 for n in 1:N]...)
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

struct UNetUpBlock{X,Y,Z}
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

struct MultiScaleConvBlock{A,B,C}
    c1::A
    c2::B
    c3::C
end
Flux.@functor MultiScaleConvBlock

function MultiScaleConvBlock(in_chs::Int, out_chs::Int; actfun, conv_layer=Conv, dims=4)
    small_kernel = tuple(ones(Int, dims - 2)...)
    medium_kernel = 3 .* small_kernel
    big_kernel = 7 .* small_kernel
    conv1a = conv_layer(medium_kernel, in_chs => out_chs, actfun; pad=SamePad())
    conv1b = conv_layer(medium_kernel, out_chs => out_chs, actfun; pad=SamePad())
    conv1 = Chain(conv1a, conv1b)
    conv2a = conv_layer(big_kernel, in_chs => out_chs, actfun; pad=SamePad())
    conv2b = conv_layer(big_kernel, out_chs => out_chs, actfun; pad=SamePad())
    conv2 = Chain(conv2a, conv2b)
    conv3 = conv_layer(small_kernel, 2*out_chs => out_chs, actfun; pad=SamePad())
    return MultiScaleConvBlock(conv1, conv2, conv3)
end

function (mscb::MultiScaleConvBlock)(x)
    x1 = mscb.c1(x)
    x2 = mscb.c2(x)
    return mscb.c3(cat(x1, x2; dims=ndims(x) - 1))
end

struct ConvBlock{F}
    chain::Chain
    actfun::F
    residual
end
Flux.trainable(c::ConvBlock) = (c.chain,)
Flux.@functor ConvBlock

"""    ConvBlock(in_chs::Int, out_chs::Int; multiscale=false, kernel=(3,3), dropout=false, activation="relu", transpose=false, residual=true, norm="batch", separable=false)

Basic building block of the `Unet`. Does dropout -> conv -> `activation` -> `norm` -> conv -> `activation` -> `norm`.

It expects data with `in_chs` channels and outputs data with `out_chs` channels.

If `residual`, the input gets passed thrugh a `1x1` convolution and added elementwise to the output of the block.

If `dropout==true`, the dropout layers are applied with a dropout probability of 50%.

If `norm=="batch"`, batchnorms will be applied, otherwise no normalization happens.

`kernel` refers to the kernel size of the convolutional layers.

`transpose` decides whether a transpose or normal convolutional layer is used.

If `separable==true` a `SeparableConv` is used instead of a normal one.

If `multiscale==true`, a `MultiScaleConvBlock` is returned instead. This overrides all settings except `in_chs`, `out_chs`, `activation`, `separable` and `transpose`.

# Examples
```julia-repl
julia> UNet.ConvBlock(10,20)
Main.UNet.ConvBlock{typeof(Main.UNet.u_relu)}(Chain(Conv((3, 3), 10 => 20, u_relu, pad=1), BatchNorm(20), identity, Conv((3, 3), 20 => 20, u_relu, pad=1), BatchNorm(20), identity), Main.UNet.u_relu, Conv((1, 1), 10 => 20))
```
"""
function ConvBlock(
    in_chs::Int,
    out_chs::Int;
    multiscale=false,
    kernel=(3, 3),
    dropout=false,
    activation="relu",
    transpose=false,
    residual=true,
    norm="batch",
    separable=false,
)
    conv_layer = Conv
    if transpose
        conv_layer = ConvTranspose
    else
        if separable
            conv_layer = SeparableConv
        end
    end

    activation_functions = Dict("relu" => u_relu, "tanh" => u_tanh, "elu" => u_elu)
    actfun = get(activation_functions, activation, identity)

    if multiscale
        return MultiScaleConvBlock(in_chs, out_chs; actfun=actfun, conv_layer=conv_layer, dims=length(kernel)+2)
    end

    conv1 = conv_layer(kernel, in_chs => out_chs, actfun; pad=1, init=Flux.glorot_normal)
    conv2 = conv_layer(kernel, out_chs => out_chs, actfun; pad=1, init=Flux.glorot_normal)

    norm1 = identity
    norm2 = identity
    if norm == "batch"
        norm1 = BatchNorm(out_chs)
        norm2 = BatchNorm(out_chs)
    end

    dropout1 = identity
    dropout2 = identity
    if dropout
        channeldim = length(kernel) + 1
        dropout1 = Dropout(0.5; dims=channeldim)
        dropout2 = Dropout(0.5; dims=channeldim)
    end
    chain = Chain(conv1, dropout1, norm1, conv2, dropout2, norm2)
    residual_func = x -> zero(eltype(x))
    if residual
        residual_func = conv_layer(ntuple(i->1, length(kernel)), in_chs=>out_chs, pad=0)
    end
    return ConvBlock(chain, actfun, residual_func)
end

function (c::ConvBlock)(x)
    y = c.chain(x)
    skip = c.residual(x)
    return y .+ skip
end

function ConvDown(
    in_chs::Int,
    out_chs::Int;
    kernel=(3, 3),
    downsample_method="maxpool",
    residual=false,
    activation="relu",
    separable=false,
    dropout=false,
    norm="batch",
    multiscale=false,
)
    down_window = ntuple(i->2, length(kernel))
    downsample_op = MaxPool(down_window)
    if downsample_method=="conv"
        downsample_op = Conv(
            down_window, in_chs => in_chs, identity; stride=2, groups=in_chs
        )
        downsample_op.weight .= 0.01 .* downsample_op.weight .+ 0.25
        downsample_op.bias .*= 0.01
    end
    conv_op = ConvBlock(
        in_chs,
        out_chs;
        kernel=kernel,
        dropout=dropout,
        activation=activation,
        residual=residual,
        separable=separable,
        norm=norm,
        multiscale=multiscale,
    )
    return Chain(downsample_op, conv_op)
end

struct Unet{T,F,R,X,Y}
    residual_block::R
    encoder::T
    decoder::F
    attention_module::X
    upsampler::Y
end

function Flux.trainable(u::Unet)
    attention = !isnothing(u.attention_module)
    residual = !isnothing(u.residual_block)
    trainables = tuple(
        u.encoder,
        u.decoder,
        [
            m for
            (m, b) in zip([u.residual_block, u.attention_module], [residual, attention]) if
            b
        ]...,
    )
    return trainables
end

Flux.@functor Unet

"""    function Unet(channels::Int=1, labels::Int=channels, dims=4; residual::Bool=false, up="nearest", down="maxpool", activation="relu", norm="batch", attention=false, depth=4, dropout=false, separable=false, final_attention=false, multiscale=false, kwargs...)

Create an U-Net, which accepts data with `channels` channels and outputs data with `labels` channels.

The total dimensionsality of the input data should be given by `dims` ( 2 or 3 spatial dimsension + 1 channel dimension + 1 batch dimension).

- `residual::Bool` : Whether to use residual double convolutions. Also whether to use a residual convolutional connection directly from the input to the output of the UNet. TODO: Separate the two residual options.

- `up::AbstractString` : Upsampling method. One of `["nearest", "tconv"]`.

- `down::AbstractString` : Downsampling method. One of `["conv", "maxpool"]`.

- `activation::AbstractString` : Activation function. Supported functions: `["relu", "tanh", "elu"]`.

- `norm::AbstractString` : Batch normalization. Either `"none"` or `"batch"`.

- `attention::Bool` : Whether to use attention gates in the skip connections.

- `depth::Int` : The length of the encoding/decoding path.

- `dropout::Bool` : Whether to use `Dropout` layers after the convolutions.

- `separable::Bool` : Whether to use separable convolution kernels.

- `final_attention::Bool` : Whether to add a layer at the end which convolves the outputs of the decoder path with a 1x1 kernel after passing a self-attention gate.

- `multiscale::Bool` : Whether to use multiscale convolution blocks, which consist of several convolution with different kernel sizes. This improves transfer learning performance, but requires a lot more computations, especially in 3D.
"""
function Unet(
    channels::Int=1,
    labels::Int=channels,
    dims=4;
    residual::Bool=false,
    up="nearest",
    down="maxpool",
    activation="relu",
    norm="batch",
    attention=false,
    depth=4,
    dropout=false,
    separable=false,
    final_attention=false,
    multiscale=false,
    kwargs...
)
    # Check that some of the options are valid
    global valid_downsampling_methods
    global valid_upsampling_methods
    @assert up in valid_upsampling_methods "Upsample method \"$up\" not in $(valid_upsampling_methods)."
    @assert down in valid_downsampling_methods "Downsampling method \"$down\" not in $(valid_downsampling_methods)."
    # Prepare a NamedTuple conv_config used for the options of the ConvBlocks.
    kernel_base = tuple(ones(Int, dims - 2)...)
    conv_kernel = kernel_base .* 3
    conv_config = (
        residual=residual,
        norm=norm,
        dropout=dropout,
        separable=separable,
        kernel=conv_kernel,
        activation=activation,
        multiscale=multiscale
    )
    # This has the same options as conv_config, except that dropout is disabled -> Used for beginning and end of UNet
    conv_config_initial = merge(conv_config, (dropout=false,))
    encoder_blocks = []
    for i in 1:depth
        second_exponent = i == depth ? i : i + 1
        c = ConvDown(
            16 * 2^i, 16 * 2^second_exponent; downsample_method=down, conv_config...
        ) # 32, 64, 128, 256, ... input channels
        push!(encoder_blocks, c)
    end

    initial_block = ConvBlock(channels, 32; conv_config...)

    residual_block = nothing
    if residual
        residual_block = ConvBlock(channels, labels; conv_config...)
    end

    attention_blocks = repeat([false], depth)
    if attention
        attention_blocks = []
        for i in 1:depth
            nrch = 16 * 2^(depth - (i - 1))
            a = AttentionBlock(nrch, nrch, nrch; dims=dims)
            push!(attention_blocks, a)
        end
    end

    up_blocks = []
    for i in 1:depth
        if up == "nearest"
            upsample_function = u_upsample_nearest
        elseif up == "tconv"
            chs = 16 * 2^(depth - (i - 1))
            upsample_function = ConvTranspose(
                tuple(2 .* ones(Int, dims - 2)...), chs => chs; stride=2, groups=chs
            )
        end
        second_index = (!final_attention && (i == depth)) ? labels : 2^(5 + depth - (i + 1))
        # The last decoding block shouldn't have dropout
        config = i == depth ? conv_config_initial : conv_config
        u = UNetUpBlock(
            upsample_function,
            attention_blocks[i],
            ConvBlock(2^(5 + depth - (i - 1)), second_index; config...),
        )
        push!(up_blocks, u)
    end

    decoder = ntuple(i -> up_blocks[i], depth)
    encoder = Chain(initial_block, encoder_blocks...)
    in_channels = sum([2^(3 + i) for i in 1:(depth + 1)])
    attention_module = if final_attention
        Chain(
            AttentionBlock(in_channels, in_channels, in_channels; dims=dims),
            Conv(kernel_base, in_channels => 1; pad=SamePad()),
        )
    else
        nothing
    end
    upsampler = dims == 4 ? upsample_bilinear : upsample_trilinear

    return Unet(residual_block, encoder, decoder, attention_module, upsampler)
end

"""    decode(ops::Tuple, ft::Tuple)

Calculate the decoding path of an UNet.

`ops` is a `Tuple` containing the decoding operations, ordered from deepest to most shallow.
`ft` is a `Tuple` that contains the activations of the encoding path, ordered from most shallow to deepest.
"""
function decode(ops::Tuple, ft::Tuple)
    up = first(ops)(ft[end], ft[end - 1])
    #= The next line looks a bit backwards, but the next `up` is calculated
       from the last two elements in ft, not the first =#
    return decode(Base.tail(ops), (ft[1:(end - 2)]..., up))..., up
end

"""    decode(::Tuple{}, ft::NTuple{1, T}) where {T}

Final `decode` operation.
"""
function decode(::Tuple{}, ft::NTuple{1,T}) where {T}
    return tuple(first(ft))
end

"""    (u::Unet)(x)

Apply a `Unet` to the input data `x`.
"""
function (u::Unet)(x)
    cs = Flux.activations(u.encoder, x)
    ups = decode(u.decoder, cs)
    up = first(ups)
    if !isnothing(u.residual_block)
        up = up .+ u.residual_block(x)
    end
    if !isnothing(u.attention_module)
        ups = (Base.tail(ups)..., cs[end])
        final_block = ups[1]
        final_size = size(final_block)[1:(end - 2)]
        for activation in ups[2:end]
            final_block = cat(
                final_block,
                u.upsampler(activation; size=final_size);
                dims=ndims(final_block) - 1,
            )
        end
        return u.attention_module(final_block)
    else
        return up
    end
end

end # module
