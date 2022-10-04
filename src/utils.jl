export readPSFs, registerPSFs
export load_data, apply_noise
export train_test_split
export gaussian
export _random_normal, _help_evaluate_loss, _ensure_existence
export my_gpu, my_cu
export train_real_gradient!
export read_yaml
export _get_default_kernel
export write_to_logfile
export _center_psfs

using MAT
using HDF5
using NDTools
using FFTW
using LinearAlgebra
using Images
using Noise
using MappedArrays
using FileIO
using CUDA
using Dates
using ProgressMeter

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

function add_noise(img)
    g_noise = randn(eltype(img), size(img)) .* (rand(eltype(img)) * 0.02 + 0.005)
    peak = rand(eltype(img)) * 4500 + 500
    img = poisson(img, peak)
    img .+= g_noise
    return img
end

function apply_noise(imgs)
    N = ndims(imgs)
    for i in 1:size(imgs, N)
        selectdim(imgs, N, i) .= add_noise(collect(selectdim(imgs, N, i)))
    end
    return imgs
end

function read_yaml(path)
    # Define dictionaries
    optimizer_dict = Dict(
        "ADAM" => Adam,
        "Descent" => Descent,
        "ADAMW" => AdamW,
        "ADAGrad" => AdaGrad,
        "ADADelta" => AdaDelta,
    )
    options = YAML.load_file(path)
    optimizer_kw = options["training"]["optimizer"]
    @assert optimizer_kw in keys(optimizer_dict) "Optimizer $optimizer_kw not defined"
    output = Dict{Symbol,Any}()
    output[:optimizer] = optimizer_dict[optimizer_kw]()
    output[:sim_dir] = options["data"]["x_path"]
    output[:truth_dir] = options["data"]["y_path"]
    output[:newsize] = tuple(options["data"]["resize_to"]...)
    loadpath = nothing
    epoch_offset = 0
    output[:load_checkpoints] = options["training"]["checkpoints"]["load_checkpoints"]
    output[:checkpoint_dir] = options["training"]["checkpoints"]["checkpoint_dir"]
    _ensure_existence(output[:checkpoint_dir])
    if output[:load_checkpoints] isa Bool && output[:load_checkpoints]
        loadpath = options["training"]["checkpoints"]["checkpoint_path"]
        epoch_offset = parse(Int, split(match(r"epoch[-][^.]*", loadpath).match, "-")[2])
        output[:checkpoint_path] = loadpath
    elseif output[:load_checkpoints] == "latest"
        # Find the most recent checkpoint in dir `checkpoint_dir`.
        # This is where the previous run should've saved checkpoints
        loadpath = output[:checkpoint_dir]
        most_recent = nothing
        most_recent_chkp = nothing
        for file in readdir(loadpath)
            if !endswith(file, ".bson")
                continue
            end
            # Separate the date in the name and format it such that it can be parsed into a `DateTime` by `tryparse`
            datestring = replace(split(file, "_loss")[1], "_" => ":")
            date = tryparse(DateTime, datestring)
            if isnothing(date)
                continue
            end
            if isnothing(most_recent) || date > most_recent
                most_recent = date
                most_recent_chkp = file
            end
        end
        if isnothing(most_recent_chkp)
            @info "No checkpoints found. Starting training from scratch"
            output[:load_checkpoints] = false
        else
            epoch_offset = parse(
                Int, split(match(r"epoch[-][^.]*", most_recent_chkp).match, "-")[2]
            )
            output[:checkpoint_path] = joinpath(loadpath, most_recent_chkp)
            output[:load_checkpoints] = true
            @info "Resuming training from $most_recent_chkp"
        end
    end
    output[:epoch_offset] = epoch_offset
    # Model parameters
    output[:depth] = options["model"]["depth"]
    output[:attention] = options["model"]["attention"]
    output[:dropout] = options["model"]["dropout"]
    output[:separable] = options["model"]["separable"]
    output[:final_attention] = options["model"]["final_attention"]
    output[:nrsamples] = options["training"]["nrsamples"]
    output[:epochs] = options["training"]["epochs"]
    output[:plot_interval] = options["training"]["plot_interval"]
    output[:plot_dir] = options["training"]["plot_path"]
    _ensure_existence(output[:plot_dir])
    output[:log_losses] = options["training"]["log_losses"]
    output[:logfile] = output[:log_losses] ? joinpath(dirname(path), "losses.log") : nothing
    output[:psfs_path] = options["training"]["psfs_path"]
    output[:psfs_key] = options["training"]["psfs_key"]
    output[:center_psfs] = options["data"]["center_psfs"]
    if output[:center_psfs]
        output[:psf_ref_index] = options["data"]["reference_index"]
    end
    output[:save_interval] = options["training"]["checkpoints"]["save_interval"]

    # Check that boolean fields have right datatype
    for field in [
        :load_checkpoints,
        :attention,
        :dropout,
        :separable,
        :final_attention,
        :log_losses,
        :center_psfs,
    ]
        temp = output[field]
        @assert temp isa Bool "$field should be a boolean, but $temp is a $(typeof(temp))."
    end
    # Int fields should be ≥ 0
    for field in
        [:epoch_offset, :depth, :nrsamples, :epochs, :plot_interval, :save_interval]
        temp = output[field]
        @assert temp isa Int "$field should be a integer, but $temp is a $(typeof(temp))."
        @assert temp ≥ zero(Int) "$field needs to be ≥ 0, but is $temp."
    end
    return output
end

"""    find_complete(nrsamples, truth_directory, simulated_directory)

Return the filenames of the first `nrsamples` files that are both in `truth_directory`
and `simulated_directory`.
"""
function find_complete(nrsamples, truth_directory, simulated_directory)
    simulated_files = readdir(simulated_directory)
    truth_files = readdir(truth_directory)
    complete_files = simulated_files ∩ truth_files
    upper_index = min(length(complete_files), nrsamples)
    return view(complete_files, 1:upper_index)
end

function _map_to_zero_one(x; T=Float32)
    min_x, max_x = T.(extrema(x))
    out_x = similar(x, T)
    out_x .= x .- min_x
    out_x .*= inv(max_x - min_x)
    return out_x
end

function load_images(
    complete_files, truth_directory, simulated_directory; newsize=(128, 128), T=Float32
)
    images_y = Array{T,4}(undef, (newsize..., 1, length(complete_files)))
    images_x = Array{T,4}(undef, (newsize..., 1, length(complete_files)))
    for (i, filename) in enumerate(complete_files)
        filepath_truth = joinpath(truth_directory, filename)
        filepath_simulated = joinpath(simulated_directory, filename)
        # TODO: Flip images along first axis?
        images_y[:, :, 1, i] .= _map_to_zero_one(imresize(load(filepath_truth), newsize))
        images_x[:, :, 1, i] .= _map_to_zero_one(
            imresize(load(filepath_simulated), newsize)
        )
    end
    return images_x, images_y
end

function load_volumes(
    complete_files,
    truth_directory,
    simulated_directory;
    newsize=(128, 128, 32),
    T=Float32,
    truth_key="gt",
    sim_key="sim",
)
    volumes_y = Array{T,5}(undef, newsize..., 1, length(complete_files))
    volumes_x = Array{T,5}(undef, newsize..., 1, length(complete_files))
    for (i, filename) in enumerate(complete_files)
        filepath_truth = joinpath(truth_directory, filename)
        filepath_simulated = joinpath(simulated_directory, filename)
        volumes_y[:, :, :, 1, i] .= _map_to_zero_one(
            imresize(readPSFs(filepath_truth, truth_key), newsize)
        )
        volumes_x[:, :, :, 1, i] .= _map_to_zero_one(
            imresize(readPSFs(filepath_simulated, sim_key), newsize)
        )
    end
    return volumes_x, volumes_y
end

function load_data(
    nrsamples, truth_directory, simulated_directory; newsize=(128, 128), T=Float32
)
    imageFileEndings = [".png", ".jpg", ".jpeg"]
    volumeFileEndings = [".mat", ".h5", ".hdf", ".hdf5", ".he5"]
    complete_files = find_complete(nrsamples, truth_directory, simulated_directory)
    if any([endswith(complete_files[1], fileEnding) for fileEnding in imageFileEndings])
        # 2D case
        x_data, y_data = load_images(
            complete_files, truth_directory, simulated_directory; newsize=newsize, T=T
        )
    elseif any([
        endswith(complete_files[1], fileEnding) for fileEnding in volumeFileEndings
    ])
        # 3D case
        x_data, y_data = load_volumes(
            complete_files, truth_directory, simulated_directory; newsize=newsize, T=T
        )
    end
    return x_data, y_data
end

"""    train_test_split(x; ratio=0.7, dim=ndims(x))

Split dataset `x` into two datasets, with the first containing `ratio` 
and the second containing `1-ratio` parts of `x`.
Splits dataset along dimension `dim` (Default is last dimension).
"""
function train_test_split(x; ratio=0.7, dim=ndims(x))
    split_ind = trunc(Int, ratio * size(x, dim))
    train = collect(selectdim(x, dim, 1:split_ind))
    test = collect(selectdim(x, dim, (1 + split_ind):size(x, dim)))
    return train, test
end

function gaussian(window_size=11, sigma=1.5; T=Float32)
    x = 1:window_size
    gauss = @. exp(-(x - ((window_size ÷ 2) + 1))^2 / (2 * sigma^2))
    return T.(gauss / sum(gauss))
end

const CUDA_functional =
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
        @tullio kernel[x, y, z] := mygaussian[x] * mygaussian[y] * mygaussian[z]
    elseif dims == 2
        kernel = mygaussian .* mygaussian'
    end
    return kernel
end

function _init_logfile(logfile)
    if !isnothing(logfile)
        open(logfile, "w") do io
            println(io, "epoch, train loss, test loss")
        end
    end
end

function write_to_logfile(logfile, epoch, train_loss, test_loss)
    if isnothing(logfile)
        return nothing
    end
    if !isfile(logfile)
        _init_logfile(logfile)
    end
    open(logfile, "a") do io
        println(io, "$(epoch), $(train_loss), $(test_loss)")
    end
end

"""    train_real_gradient!(loss, ps, data, opt)

Same as `Flux.train!` but with real gradient.
"""
function train_real_gradient!(loss, ps, data, opt; batch_size=2)
    # Zygote calculates a complex gradient, even though this is mapping  real -> real.
    # Might have to do with fft and incomplete Wirtinger derivatives? Anyway, only
    # use the real part of the gradient
    @showprogress "Epoch progress:" for part in Iterators.partition(data, batch_size)
        try
            part = my_cu(part)
            gs = Flux.gradient(ps) do
                l = 0
                for d in part
                    l = l + loss(Flux.Optimise.batchmemaybe(d)...)
                end
                l
            end
            part = nothing
            Flux.update!(opt, ps, real.(gs))
        catch ex
            if ex isa Flux.Optimise.StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end

function _center_psfs(psfs, center, ref_index)
    if !center
        return psfs
    end
    ref_index = if ref_index == -1
        size(psfs)[end] ÷ 2 + 1
    else
        ref_index
    end
    psfs, _ = registerPSFs(psfs, collect(selectdim(psfs, ndims(psfs), ref_index)))
    return psfs
end

pretty_summarysize(x) = Base.format_bytes(Base.summarysize(x))

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
        return psfs
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

    yi_reg = similar(stack, size(stack)...)
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
    im_reg = similar(stack_dct, Ns...)
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

    if N == 4 && maximum(abs.(si[3, :])) > zero(eltype(si))
        for ind in good_indices
            selected_stack = selectdim(stack, ND, ind)
            linshift!(im_reg, selected_stack, si[:, ind])
            selectdim(yi_reg, ND, ind) .= im_reg
        end
    end

    # Populate yi_reg
    for ind in good_indices
        circshift!(im_reg, selectdim(stack, ND, ind), si[:, ind])
        selectdim(yi_reg, ND, ind) .= im_reg
    end

    return collect(selectdim(yi_reg, ND, 1:(good_count - 1))), si
end
