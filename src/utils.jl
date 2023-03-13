export prepare_psfs
export load_data, apply_noise
export train_validation_split
export prepare_data
export prepare_model!
export gaussian
export _random_normal, _help_evaluate_loss, _ensure_existence
export my_gpu, my_cu
export train_real_gradient!
export read_yaml
export _get_default_kernel
export write_to_logfile
export _center_psfs
export Settings
export get_load_data_settings

using MAT
using HDF5
using NDTools
using FFTW
using Flux
using LinearAlgebra
using Images
using Noise
using MappedArrays
using FileIO
using CUDA
using Dates
using ProgressMeter
using Statistics

# Hardcoded mappings between the config yaml fields and internally used symbols
const data_dict = Dict(
    :sim_dir => "x_path",
    :truth_dir => "y_path",
    :nrsamples => "nrsamples",
    :newsize => "resize_to",
    :center_psfs => "center_psfs",
    :psf_ref_index => "reference_index",
    :psfs_path => "psfs_path",
    :psfs_key => "psfs_key",
    :snr => "SNR",
    :channels => "channels",
)
const model_dict = Dict(
    :depth => "depth",
    :attention => "attention",
    :dropout => "dropout",
    :separable => "separable",
    :final_attention => "final_attention",
    :multiscale => "multiscale",
    :deconv => "deconv",
)
const training_dict = Dict(
    :epochs => "epochs",
    :optimizer => "optimizer",
    :plot_interval => "plot_interval",
    :plot_dir => "plot_path",
    :log_losses => "log_losses",
    :early_stopping => "early_stopping",
    :batchsize => "batchsize",
    :weight_decay => "weight_decay",
)
const checkpoint_dict = Dict(
    :load_checkpoints => "load_checkpoints",
    :checkpoint_dir => "checkpoint_dir",
    :checkpoint_path => "checkpoint_path",
    :save_interval => "save_interval",
)
const optimizer_dict = Dict(
    "ADAM" => Adam,
    "Descent" => Descent,
    "ADAMW" => AdamW,
    "ADAGrad" => AdaGrad,
    "ADADelta" => AdaDelta,
)

#=
function load_dataset(
    nrsamples, truth_directory, simulated_directory, nd=2; newsize=(128, 128)
)
    files = find_complete(nrsamples, truth_directory, simulated_directory)
    if nd == 2
        loader =
            x -> my_gpu(
                add_noise(
                    load_images(x, truth_directory, simulated_directory; newsize=newsize)
                ),
            )
    elseif nd == 3
        loader =
            x -> my_gpu(
                add_noise(
                    loadvolumes(x, truth_directory, simulated_directory; newsize=newsize),
                ),
            )
    end
    return mappedarray(loader, files)
end
=#

"""    add_noise(img::AbstractArray{T}; SNR=70) where {T}

Add a mix of gaussian and poissionian noise to an image. 

Returns the noisy image.

A signal-to-noise ration `SNR` can be specified. 

The SNR is defined as the contrast of the image divided by the standard deviation of the noise.

In the special case of this package, the contrast is always assumed to be unity.

The employed Poisson scaling `λ` is related to the 
gaussian standard deviation `σ` by `σ = 1/√λ`.
"""
function add_noise(img::AbstractArray{T}; SNR=70) where {T}
    # Define the gaussian and possion noise parameters such that the resulting images has the specified SNR
    μ = mean(img)
    if iszero(μ)
        return img
    end
    λ = SNR^2 * (1/μ + 1/μ^2)
    σ = inv(√λ)
    # Apply the noise
    gaussian_noise = randn(T, size(img)) .* σ
    poisson_image = poisson(img, λ * maximum(img))
    return poisson_image .+ gaussian_noise
end

function apply_noise(imgs; SNR=70)
    N = ndims(imgs)
    for i in 1:size(imgs, N)
        selectdim(imgs, N, i) .= add_noise(collect(selectdim(imgs, N, i)); SNR=SNR)
    end
    return imgs
end

struct Settings
    data::Dict{Symbol,Any}
    model::Dict{Symbol,Any}
    training::Dict{Symbol,Any}
    checkpoints::Dict{Symbol,Any}
end

function Settings(path)
    in = YAML.load_file(path)
    data = in["data"]
    my_data = Dict{Symbol,Any}()
    for (key, value) in data_dict
        my_data[key] = data[value]
    end
    my_data = process_data_dict(my_data)

    model = in["model"]
    my_model = Dict{Symbol,Any}()
    for (key, value) in model_dict
        my_model[key] = model[value]
    end
    my_model = process_model_dict(my_model)

    training = in["training"]
    my_training = Dict{Symbol,Any}()
    for (key, value) in training_dict
        my_training[key] = training[value]
    end
    my_training = process_training_dict(my_training; path=path)

    checkpoint = in["checkpoints"]
    my_checkpoints = Dict{Symbol,Any}()
    for (key, value) in checkpoint_dict
        my_checkpoints[key] = checkpoint[value]
    end
    my_checkpoints = process_checkpoints_dict(my_checkpoints)
    return Settings(my_data, my_model, my_training, my_checkpoints)
end

function get_load_data_settings(s::Settings)
    return s.data[:nrsamples], s.data[:truth_dir], s.data[:sim_dir], s.data[:newsize], s.data[:channels]
end

function check_types(type_dict, value_dict)
    for (key, value) in type_dict
        temp = value_dict[key]
        @assert temp isa value "$key should be a $value, but $temp is a $(typeof(temp))."
    end
end

function process_data_dict(my_data)
    type_dict = Dict(
        :sim_dir => String,
        :truth_dir => String,
        :nrsamples => Int,
        :center_psfs => Bool,
        :psf_ref_index => Int,
        :psfs_path => String,
        :psfs_key => String,
        :snr => Int,
        :channels => Int,
    )
    check_types(type_dict, my_data)
    my_data[:newsize] = tuple(my_data[:newsize]...)
    return my_data
end

function process_model_dict(my_model)
    type_dict = Dict(
        :depth => Int,
        :attention => Bool,
        :dropout => Bool,
        :separable => Bool,
        :final_attention => Bool,
        :multiscale => Bool,
        :deconv => String,
    )
    check_types(type_dict, my_model)
    valid_deconv = ["rl", "wiener", "rl_flfm"]
    @assert my_model[:deconv] in valid_deconv "deconv has to be one of $valid_deconv, but is $(my_model[:deconv])."
    return my_model
end

function process_training_dict(my_training; kwargs...)
    type_dict = Dict(
        :epochs => Int,
        :plot_interval => Int,
        :plot_dir => String,
        :log_losses => Bool,
        :early_stopping => Int,
        :batchsize => Int,
        :weight_decay => Float64,
    )
    check_types(type_dict, my_training)
    optimizer_kw = my_training[:optimizer]
    @assert optimizer_kw in keys(optimizer_dict) "Optimizer $optimizer_kw not implemented"
    @assert my_training[:early_stopping] >= 0 "early_stopping must be >= 0"
    @assert my_training[:batchsize] > 0 "batchsize must be > 0"
    @assert my_training[:weight_decay] >= 0.0 "weight_decay must be positive"

    opt = optimizer_dict[optimizer_kw]()
    my_training[:optimizer] = if iszero(my_training[:weight_decay])
        opt
    else
        Flux.Optimiser(WeightDecay(my_training[:weight_decay]), opt)
    end
    _ensure_existence(my_training[:plot_dir])
    my_training[:logfile] =
        my_training[:log_losses] ? joinpath(dirname(kwargs[:path]), "losses.log") : nothing
    return my_training
end

function process_checkpoints_dict(my_checkpoints)
    type_dict = Dict(:checkpoint_dir => String, :save_interval => Int)
    check_types(type_dict, my_checkpoints)
    @assert my_checkpoints[:load_checkpoints] in [true, false, "latest"] "load_checkpoints needs to be one of [true, false, \"latest\"]."
    if my_checkpoints[:load_checkpoints] isa Bool && my_checkpoints[:load_checkpoints]
        my_checkpoints[:epoch_offset] = parse_epoch(my_checkpoints[:checkpoint_path])
    elseif my_checkpoints[:load_checkpoints] == "latest"
        epoch_offset, load_checkpoints, checkpoint_path = find_most_recent_checkpoint(
            my_checkpoints[:checkpoint_dir]
        )
        my_checkpoints[:epoch_offset] = epoch_offset
        my_checkpoints[:load_checkpoints] = load_checkpoints
        my_checkpoints[:checkpoint_path] = checkpoint_path
    else
        my_checkpoints[:epoch_offset] = 0
        delete!(my_checkpoints, :checkpoint_path)
    end
    return my_checkpoints
end

function find_most_recent_checkpoint(path)
    # Find the most recent checkpoint in dir `checkpoint_dir`.
    # This is where the previous run should've saved checkpoints
    most_recent = nothing
    most_recent_chkp = nothing
    checkpoint_path = nothing
    load_checkpoints = false
    epoch_offset = 0
    if isdir(path)
        for file in readdir(path)
            date = parse_date(file)
            if isnothing(date)
                continue
            end
            if isnothing(most_recent) || date > most_recent
                most_recent = date
                most_recent_chkp = file
            end
        end
        if isnothing(most_recent_chkp)
            @info "No checkpoints found. Starting training from scratch."
            load_checkpoints = false
        else
            epoch_offset = parse_epoch(most_recent_chkp)
            checkpoint_path = joinpath(path, most_recent_chkp)
            load_checkpoints = true
            @info "Resuming training from $most_recent_chkp."
        end
    end
    return epoch_offset, load_checkpoints, checkpoint_path
end

function parse_epoch(checkpoint_path)
    return parse(Int, split(match(r"epoch[-][^.]*", checkpoint_path).match, "-")[2])
end

function parse_date(checkpoint_path)
    if !endswith(checkpoint_path, ".bson")
        return nothing
    end
    # Separate the date in the name and format it such that it can be parsed into a `DateTime` by `tryparse`
    datestring = replace(split(checkpoint_path, "_loss")[1], "_" => ":")
    date = tryparse(DateTime, datestring)
    return date # Could be a DateTime or nothing
end

"""    remove_file_extension(filename)

Return the part of a string `filename` that occurs before the last '.'.
"""
function remove_file_extension(filename)
    return filename[1:(findlast(isequal('.'), filename) - 1)]
end

"""    find_complete(nrsamples, truth_directory, simulated_directory)

Return the filenames of the first `nrsamples` files that are both in `truth_directory`
and `simulated_directory`.
"""
function find_complete(nrsamples, truth_directory, simulated_directory)
    simulated_files = readdir(simulated_directory)
    truth_files = readdir(truth_directory)
    complete_files = [
        (t, s) for (t, s) in zip(truth_files, simulated_files) if
        remove_file_extension(t) == remove_file_extension(s)
    ]
    upper_index = min(length(complete_files), nrsamples)
    valid_names = complete_files[1:upper_index]
    return first.(valid_names), last.(valid_names)
end

function _map_to_zero_one(img, min_x, max_x)
    out = img .- min_x
    out .*= inv(max_x - min_x)
    return out
end

function _map_to_zero_one(img)
    return _map_to_zero_one(img, extrema(img)...)
end

function _img_to_rgb(img, T=Float32)
    img_r = red.(img)
    img_g = green.(img)
    img_b = blue.(img)
    return T.(cat(img_r, img_g, img_b; dims=3))
end

function _rgb_to_img(img)
    img = _map_to_zero_one(img, extrema(img)...)
    img_r = selectdim(img, 3, 1)
    img_g = selectdim(img, 3, 2)
    img_b = selectdim(img, 3, 3)
    return colorview(RGB, img_r, img_g, img_b)
end

function load_images(complete_files, directory; newsize=(128, 128), T=Float32, channels=1)
    images = Array{T,4}(undef, (newsize..., channels, length(complete_files)))
    to_channels = channels==1 ? identity : x -> _img_to_rgb(x, T)
    for (i, filename) in enumerate(complete_files)
        filepath = joinpath(directory, filename)
        img = _map_to_zero_one(to_channels(imresize(load(filepath), newsize)))
        images[:, :, :, i] .= any(isnan.(img)) ? 0.0 : img
    end
    return images
end

function load_volumes(
    complete_files, directory; newsize=(128, 128, 32), T=Float32, key="gt"
)
    volumes = Array{T,5}(undef, newsize..., 1, length(complete_files))
    for (i, filename) in enumerate(complete_files)
        filepath = joinpath(directory, filename)
        volumes[:, :, :, 1, i] .= _map_to_zero_one(
            imresize(readPSFs(filepath, key)[1], newsize)
        )
    end
    return volumes
end

"""    is_image(filename)

Returns `true` if `filename` ends with an image extension.
"""
function is_image(filename)
    imageFileEndings = [".png", ".jpg", ".jpeg"]
    return any([endswith(filename, fileEnding) for fileEnding in imageFileEndings])
end

"""    is_volume(filename)

Returns `true` if `filename` ends with a `mat` or `HDF5` extension.
"""
function is_volume(filename)
    volumeFileEndings = [".mat", ".h5", ".hdf", ".hdf5", ".he5"]
    return any([endswith(filename, fileEnding) for fileEnding in volumeFileEndings])
end

function load_data(settings::Settings; T=Float32)
    nrsamples, truth_directory, simulated_directory, newsize, channels = get_load_data_settings(
        settings
    )
    complete_files_truth, complete_files_sim = find_complete(
        nrsamples, truth_directory, simulated_directory
    )
    if all(is_image.([complete_files_sim[1], complete_files_truth[1]])) &&
        settings.model[:deconv] != "rl_flfm"
        # 2D => 2D deconvolution with Wiener or RL deconvolution
        x_data = load_images(complete_files_sim, simulated_directory; newsize=newsize, T=T, channels=channels)
        y_data = load_images(complete_files_truth, truth_directory; newsize=newsize, T=T, channels=channels)
    elseif all(is_image.([complete_files_sim[1], complete_files_truth[1]])) &&
        settings.model[:deconv] == "rl_flfm"
        # 2D => 2D deconvolution with RL deconvolution for FLFM. This deconvolution needs 3D input and output, so if
        # 2D images are loaded, make the pseudo-3D by adding a singleton z-dimension.
        x_data = load_images(complete_files_sim, simulated_directory; newsize=newsize, T=T)
        y_data = load_images(complete_files_truth, truth_directory; newsize=newsize, T=T)
        x_data = reshape(x_data, size(x_data)[1:2]..., 1, size(x_data)[3:end]...)
        y_data = reshape(y_data, size(y_data)[1:2]..., 1, size(y_data)[3:end]...)
    elseif all(is_volume.([complete_files_sim[1], complete_files_truth[1]]))
        x_data = load_volumes(
            complete_files_sim, simulated_directory; newsize=newsize, T=T, key="sim"
        )
        y_data = load_volumes(
            complete_files_truth, truth_directory; newsize=newsize, T=T, key="gt"
        )
    elseif is_volume(complete_files_truth[1]) && is_image(complete_files_sim[1])
        # 2D => 3D reconstruction
        y_data = load_volumes(
            complete_files_truth, truth_directory; newsize=newsize, T=T, key="gt"
        )
        x_data = load_images(
            complete_files_sim, simulated_directory; newsize=newsize[1:2], T=T
        )
        # x_data needs to be reshaped by adding a singleton z-dimension, so broadcasting works in the RL_FLFM layer
        x_data = reshape(x_data, size(x_data)[1:2]..., 1, size(x_data)[3:end]...)
    else
        error("Unknown imaging modality. Supported modalities: 3D=>3D, 2D=>2D and 3D=>2D")
    end

    return x_data, y_data
end

"""    train_validation_split(x; ratio=0.7, dim=ndims(x))

Split dataset `x` into two datasets, with the first containing `ratio` 
and the second containing `1-ratio` parts of `x`.
Splits dataset along dimension `dim` (Default is last dimension).
"""
function train_validation_split(x; ratio=0.7, dim=ndims(x))
    split_ind = trunc(Int, ratio * size(x, dim))
    train = collect(selectdim(x, dim, 1:split_ind))
    validation = collect(selectdim(x, dim, (1 + split_ind):size(x, dim)))
    return train, validation
end

"""    prepare_model!(settings::Settings)

Either load a previously loaded model or initialize a new one.

Side effect: `settings.training[:optimizer]` gets set to the loaded optimizer
if a model is loaded from a checkpoint.
"""
function prepare_model!(settings::Settings)
    if !settings.checkpoints[:load_checkpoints]
        psfs = prepare_psfs(settings)
        psfs = my_gpu(psfs)
        model = my_gpu(make_model(psfs, settings.model; channels=settings.data[:channels]))
    else
        model, optimizer = my_gpu(load_model(settings.checkpoints[:checkpoint_path]))
        settings.training[:optimizer] = optimizer
    end
    return model
end

function prepare_data(settings::Settings; T=Float32)
    x_data, y_data = load_data(settings; T=T)
    SNR = settings.data[:snr]
    x_data = apply_noise(x_data; SNR=SNR)
    #x_data = x_data .* convert(eltype(x_data), 2) .- one(eltype(x_data))
    #y_data = y_data .* convert(eltype(y_data), 2) .- one(eltype(y_data))
    train_x, validation_x = train_validation_split(x_data)
    train_y, validation_y = train_validation_split(y_data)
    return train_x, train_y, validation_x, validation_y
end

function gaussian(window_size=11, sigma=1.5; T=Float32)
    x = 1:window_size
    gauss = @. exp(-(x - ((window_size ÷ 2) + 1))^2 / (2 * sigma^2))
    return T.(gauss / sum(gauss))
end

const CUDA_functional::Bool =
    CUDA.functional() &&
    any([CUDA.capability(dev) for dev in CUDA.devices()] .>= VersionNumber(3, 5, 0))

function show_cuda_capability()
    global CUDA_functional
    if CUDA_functional
        @info "Running on GPU"
    else
        @info "Running on CPU"
    end
end

function my_cu(x)
    global CUDA_functional
    if CUDA_functional
        return cu(x)
    end
    return x
end

function my_gpu(x)
    global CUDA_functional
    if CUDA_functional
        return gpu(x)
    end
    return x
end

function _help_evaluate_loss(data, loss)
    losses = zeros(length(data))
    @showprogress "Evaluation progress:" for (i, d) in enumerate(data)
        d = my_cu(d)
        losses[i] = loss(Flux.Optimise.batchmemaybe(d)...)
        d = nothing
    end
    return first.(losses)
end

"""    _ensure_existence(dir)

If directory `dir` does not exist, create it (including intermediate directories).
"""
function _ensure_existence(dir)
    return isdir(dir) || mkpath(dir)
end

"""    _get_default_kernel(dims; T=Float32)

Return a `dims`-dimensional gaussian with sidelength 11 and σ=1.5 with `eltype` `T`.
"""
function _get_default_kernel(dims; T=Float32)
    mygaussian = gaussian(; T=T)
    if dims == 3
        N = length(mygaussian)
        kernel = Array{T,3}(undef, N, N, N)
        for c in CartesianIndices(kernel)
            kernel[c.I...] = mygaussian[c.I[1]] .* mygaussian[c.I[2]] * mygaussian[c.I[3]]
        end
    elseif dims == 2
        kernel = mygaussian .* mygaussian'
    end
    return kernel
end

function _init_logfile(logfile)
    if !isnothing(logfile)
        open(logfile, "w") do io
            println(io, "epoch, train loss, validation loss")
        end
    end
end

function write_to_logfile(logfile, epoch, train_loss, validation_loss)
    if isnothing(logfile)
        return nothing
    end
    if !isfile(logfile)
        _init_logfile(logfile)
    end
    open(logfile, "a") do io
        println(io, "$(epoch), $(train_loss), $(validation_loss)")
    end
end

"""    train_real_gradient!(loss, ps, data, opt)

Same as `Flux.train!` but with real gradient.
"""
function train_real_gradient!(loss, ps, data, opt; batch_size=2)
    # Zygote calculates a complex gradient, even though this is mapping  real -> real.
    # Might have to do with fft and incomplete Wirtinger derivatives? Anyway, only
    # use the real part of the gradient
    p = Progress(length(Iterators.partition(data, batch_size)), 1, "Epoch progress:")
    l = 0
    for part in Iterators.partition(data, batch_size)
        try
            part = my_cu.(part)
            gs = Flux.gradient(ps) do
                l = 0
                for d in part
                    l = l + loss(Flux.Optimise.batchmemaybe(d)...)
                end
                l
            end
            part = nothing
            Flux.update!(opt, ps, real.(gs))
            next!(p; showvalues=[(:loss, l)])
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            else
                rethrow(ex)
            end
        end
    end
    return finish!(p)
end

function _center_psfs(psfs, center, ref_index, positions, channels)
    if !center
        return psfs
    end
    ref_index = if ref_index == -1
        size(psfs)[end] ÷ 2 + 1
    else
        ref_index
    end
    spatial_dims = channels == 1 ? ndims(psfs) - 1 : ndims(psfs) - 2
    channel_dim = ndims(psfs) - 1
    psfs = channels == 1 ? reshape(psfs, size(psfs)[1:spatial_dims]..., 1, size(psfs, ndims(psfs))) : psfs
    psfs_registration = selectdim(sqrt.(sum(abs2, psfs; dims=channel_dim)), channel_dim, 1)
    _, shifts = registerPSFs(
                    psfs_registration, collect(selectdim(psfs_registration, ndims(psfs_registration), ref_index))
                )
    if isnothing(positions)
        for i in axes(psfs, channel_dim)
            channel_psf = selectdim(psfs, channel_dim, i)
            selectdim(psfs, channel_dim, i) .= shift_psfs(channel_psf, shifts)
        end
    else
        center_pos = size(psfs)[1:spatial_dims] .÷ 2 .+ 1
        shifts = 1 .* (center_pos .- positions)
        psfs = shift_psfs(psfs, shifts)
    end
    return psfs
end

pretty_summarysize(x) = Base.format_bytes(Base.summarysize(x))

function prepare_psfs(settings::Settings; T=Float32)
    uncentered_psfs, positions = readPSFs(
        settings.data[:psfs_path], settings.data[:psfs_key]
    )
    channels = settings.data[:channels]
    psfs = _center_psfs(
        uncentered_psfs,
        settings.data[:center_psfs],
        settings.data[:psf_ref_index],
        positions,
        channels
    )
    dims = length(settings.data[:newsize])
    nrPSFs = size(psfs)[end]
    resized_psfs = Array{T,dims + 2}(undef, settings.data[:newsize]..., channels, nrPSFs)
    for i in 1:nrPSFs
        selectdim(resized_psfs, dims + 2, i) .= imresize(
            collect(selectdim(psfs, dims + 2, i)), (settings.data[:newsize]..., channels)
        )
    end
    psfs = resized_psfs
    return psfs
end

#= readPSFs and registerPSFs should eventually be imported from SpatiallyVaryingConvolution=#

function readPSFs(path::String, key::String)
    hdf5FileEndings = [".h5", ".hdf", ".hdf5", ".he5"]
    matFileEndings = [".mat"]
    if any([endswith(path, fileEnding) for fileEnding in matFileEndings])
        file = matopen(path)
    elseif any([endswith(path, fileEnding) for fileEnding in hdf5FileEndings])
        file = h5open(path, "r")
    end

    if haskey(file, key)
        psfs = read(file, key)
        if haskey(file, "positions")
            positions = read(file, "positions")
            return psfs, positions
        end
        return psfs, nothing
    end
end

"""    padND(x, n)

Pad `x` along the first `n` dimensions with `0` to twice its size.
"""
function padND(x, n)
    return select_region(x; new_size=2 .* size(x)[1:n], pad_value=zero(eltype(x)))
end

function linshift!(
    dest::AbstractArray{T,N},
    src::AbstractArray{T,N},
    shifts::AbstractArray{F,1};
    filler=zero(T),
) where {T,F,N}
    myshifts = ntuple(i -> shifts[i], length(shifts))
    for ind in CartesianIndices(dest)
        shifted_ind = ind.I .- myshifts
        value = filler
        if !(
            any(shifted_ind .<= zero(eltype(shifted_ind))) || any(shifted_ind .> size(src))
        )
            value = src[shifted_ind...]
        end
        dest[ind.I...] = value
    end
end

"""
    registerPSFs(stack, ref_im)
Find the shift between each PSF in `stack` and the reference PSF in `ref_im`
and return the aligned PSFs and their shifts.
 
If `ref_im` has size `(Ny, Nx)`/`(Ny, Nx, Nz)`, `stack` should have size
 `(Ny, Nx, nrPSFs)`/`(Ny, Nx, Nz, nrPSFs)`.
"""
function registerPSFs(stack::AbstractArray{T,N}, ref_im) where {T,N}
    @assert N in [3, 4] "stack needs to be a 3d/4d array but was $(N)d"
    ND = ndims(stack)
    Ns = size(stack)[1:(end - 1)]
    ps = Ns # Relative centers of all correlations
    M = size(stack)[end]
    pad_function = x -> padND(x, ND - 1)

    function crossCorr(
        x::AbstractArray{Complex{T}},
        y::AbstractArray{Complex{T}},
        iplan::AbstractFFTs.ScaledPlan,
    )
        return fftshift(iplan * (x .* y))
    end

    function norm(x)
        return sqrt(sum(abs2.(x)))
    end

    stack_dct = copy(stack)
    ref_norm = norm(ref_im) # norm of ref_im

    # Normalize the stack
    norms = map(norm, eachslice(stack_dct; dims=ND))
    norms = reshape(norms, ones(Int, ND - 1)..., length(norms))
    stack_dct ./= norms
    ref_im ./= ref_norm

    si = similar(ref_im, Int, ND - 1, M)
    good_indices = []
    # Do FFT registration
    good_count = 1
    dummy_for_plan = similar(stack_dct, (2 .* Ns)...)
    plan = plan_rfft(dummy_for_plan; flags=FFTW.MEASURE)
    iplan = inv(plan)
    pre_comp_ref_im = conj.(plan * (pad_function(ref_im)))
    ft_stack = similar(stack_dct, Complex{T}, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...)
    padded_stack_dct = pad_function(stack_dct)
    for m in 1:M
        mul!(ft_stack, plan, selectdim(padded_stack_dct, ND, m))
        corr_im = crossCorr(ft_stack, pre_comp_ref_im, iplan)
        max_value, max_location = findmax(corr_im)
        if max_value < 0.01
            println("Image $m has poor quality. Skipping")
            continue
        end

        si[:, good_count] .= 1 .+ ps .- max_location.I
        push!(good_indices, m)
        good_count += 1
    end
    yi_reg = shift_psfs(stack, si, good_indices)
    return collect(selectdim(yi_reg, ND, 1:(good_count - 1))), si
end

"""    shift_psfs(stack::AbstractArray{T,N}, shift_indices, good_indices=1:size(stack, N)) where {T,N}
Shift each measurement image/volume in `stack` by the x-y-z-shifts given in `shift_indices` (`size(shift_indices)=(N, size(stack, N))`).
Only the measurements indexed by `good_indices` are considered.
"""
function shift_psfs(
    stack::AbstractArray{T,N}, shift_indices, good_indices=1:size(stack, N)
) where {T,N}
    # Output destination
    yi_reg = similar(stack)
    # Temporary shifting destination
    im_reg = similar(stack, size(stack)[1:(end - 1)])
    #Populate yi_reg
    # In 3D, with z-shift, do a linear shift (without wrap-around)
    if N == 4 && maximum(abs.(shift_indices[3, :])) > zero(eltype(shift_indices))
        for ind in good_indices
            selected_stack = selectdim(stack, N, ind)
            linshift!(im_reg, selected_stack, shift_indices[:, ind])
            selectdim(yi_reg, N, ind) .= im_reg
        end
    else
        # Else, a circshift should be good enough
        for ind in good_indices
            circshift!(im_reg, selectdim(stack, N, ind), shift_indices[:, ind])
            selectdim(yi_reg, N, ind) .= im_reg
        end
    end
    return yi_reg
end
