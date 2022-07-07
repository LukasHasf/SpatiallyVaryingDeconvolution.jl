export readPSFs, registerPSFs
export load_data, applynoise
export train_test_split
export gaussian
export _random_normal, _help_evaluate_loss, _ensure_existence
export my_gpu, my_cu
export train_real_gradient!
export read_yaml

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
using ProgressMeter

function load_dataset(
    nrsamples, truth_directory, simulated_directory, nd=2; newsize=(128, 128)
)
    files = find_complete(nrsamples, truth_directory, simulated_directory)
    if nd == 2
        loader =
            x -> my_gpu(
                addnoise(
                    loadimages(x, truth_directory, simulated_directory; newsize=newsize)
                ),
            )
    elseif nd == 3
        loader =
            x -> my_gpu(
                addnoise(
                    loadvolumes(x, truth_directory, simulated_directory; newsize=newsize),
                ),
            )
    end
    return mappedarray(loader, files)
end

function addnoise(img)
    g_noise = randn(eltype(img), size(img)) .* (rand(eltype(img)) * 0.02 + 0.005)
    peak = rand(eltype(img)) * 4500 + 500
    img = poisson(img, peak)
    img .+= g_noise
    return img
end

function applynoise(imgs)
    N = ndims(imgs)
    for i in 1:size(imgs, N)
        selectdim(imgs, N, i) .= addnoise(collect(selectdim(imgs, N, i)))
    end
    return imgs
end

function read_yaml(path)
    # Define dictionaries
    optimizer_dict = Dict(
        "ADAM" => Flux.Optimise.ADAM,
        "Descent" => Flux.Optimise.Descent,
        "ADAMW" => Flux.Optimise.ADAMW,
        "ADAGrad" => Flux.Optimise.ADAGrad,
        "ADADelta" => Flux.Optimise.ADADelta,
    )
    options = YAML.load_file(path)
    optimizer_kw = options["training"]["optimizer"]
    @assert optimizer_kw in keys(optimizer_dict) "Optimizer $optimizer_kw not defined"
    output = Dict()
    output["optimizer"] = optimizer_dict[optimizer_kw]()
    output["sim dir"] = options["data"]["x_path"]
    output["truth dir"] = options["data"]["y_path"]
    output["newsize"] = tuple(options["data"]["resize_to"]...)
    loadpath = nothing
    epoch_offset = 0
    output["load checkpoints"] = options["training"]["checkpoints"]["load_checkpoints"]
    if output["load checkpoints"] 
        loadpath = options["training"]["checkpoints"]["checkpoint_path"]
        epoch_offset = parse(Int, split(match(r"epoch[-][^.]*", loadpath).match, "-")[2])
        output["checkpoint path"] = loadpath
    end
    output["epoch offset"] = epoch_offset
    # Model parameters
    output["depth"] = options["model"]["depth"]
    output["attention"] = options["model"]["attention"]
    output["dropout"] = options["model"]["dropout"]
    output["nrsamples"] = options["training"]["nrsamples"]
    output["epochs"] = options["training"]["epochs"]
    output["plot interval"] = options["training"]["plot_interval"]
    output["plot dir"] =  options["training"]["plot_path"]
    _ensure_existence(output["plot dir"])
    output["log losses"] = options["training"]["log_losses"]
    logfile =  output["log losses"] ? joinpath(dirname(path), "losses.log") : nothing
    if output["log losses"]
        output["logfile"] = logfile
    end
    output["psfs path"] = options["training"]["psfs_path"]
    output["psfs key"] = options["training"]["psfs_key"]
    output["center psfs"] = options["data"]["center_psfs"]
    if output["center psfs"]
        output["psf ref index"] = options["data"]["reference_index"]
    end
    output["checkpoint dir"] = options["training"]["checkpoints"]["checkpoint_dir"]
    _ensure_existence(output["checkpoint dir"])
    output["save interval"] = options["training"]["checkpoints"]["save_interval"]
    return output
end

"""    find_complete(nrsamples, truth_directory, simulated_directory)

Return the filenames of the first `nrsamples` files that are both in `truth_directory`
and `simulated_directory`.
"""
function find_complete(nrsamples, truth_directory, simulated_directory)
    complete_files = Array{String,1}(undef, nrsamples)
    counter = 0
    simulated_files = readdir(simulated_directory)
    for filename in readdir(truth_directory)
        if filename in simulated_files
            complete_files[counter + 1] = filename
            counter += 1
        end
        if counter == nrsamples
            return complete_files
        end
    end
    if counter < nrsamples
        return view(complete_files, 1:counter)
    end
end

function loadimages(
    complete_files, truth_directory, simulated_directory; newsize=(128, 128), T=Float32
)
    images_y = Array{T,4}(undef, (newsize..., 1, length(complete_files)))
    images_x = Array{T,4}(undef, (newsize..., 1, length(complete_files)))
    for (i, filename) in enumerate(complete_files)
        filepath_truth = joinpath(truth_directory, filename)
        filepath_simulated = joinpath(simulated_directory, filename)
        # TODO: Flip images along first axis?
        images_y[:, :, 1, i] .= imresize(load(filepath_truth), newsize)
        images_x[:, :, 1, i] .= imresize(load(filepath_simulated), newsize)
    end
    return images_x, images_y
end

function loadvolumes(
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
        volumes_y[:, :, :, 1, i] .= imresize(readPSFs(filepath_truth, truth_key), newsize)
        volumes_x[:, :, :, 1, i] .= imresize(readPSFs(filepath_simulated, sim_key), newsize)
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
        x_data, y_data = loadimages(
            complete_files, truth_directory, simulated_directory; newsize=newsize, T=T
        )
    elseif any([
        endswith(complete_files[1], fileEnding) for fileEnding in volumeFileEndings
    ])
        # 3D case
        x_data, y_data = loadvolumes(
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

function gaussian(window_size, sigma)
    x = 1:window_size
    gauss = @. exp(-(x - ((window_size ÷ 2) + 1))^2 / (2 * sigma^2))
    return gauss / sum(gauss)
end

const CUDA_functional =
    CUDA.functional() &&
    any([CUDA.capability(dev) for dev in CUDA.devices()] .>= VersionNumber(3, 5, 0))

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

function _help_evaluate_loss(arr_x, arr_y, index::Union{Int,UnitRange}, loss_fn)
    tmp_x = copy(selectdim(arr_x, ndims(arr_x), index))
    tmp_y = copy(selectdim(arr_y, ndims(arr_y), index))
    if index isa Int
        tmp_x = reshape(tmp_x, size(tmp_x)..., 1)
        tmp_y = reshape(tmp_y, size(tmp_y)..., 1)
    end
    return loss_fn(my_gpu(tmp_x), my_gpu(tmp_y))
end

"""    _ensure_existence(dir)

If directory `dir` does not exist, create it (including intermediate directories).
"""
function _ensure_existence(dir)
    if !isdir(dir)
        mkpath(dir)
    end
end

"""    train_real_gradient!(loss, ps, data, opt)

Same as `Flux.train!` but with real gradient.
"""
function train_real_gradient!(loss, ps, data, opt)
    # Zygote calculates a complex gradient, even though this is mapping  real -> real.
    # Might have to do with fft and incomplete Wirtinger derivatives? Anyway, only
    # use the real part of the gradient
    @showprogress "Epoch progress:" for (i, d) in enumerate(data)
        try
            d = my_cu(d)
            gs = Flux.gradient(ps) do
                loss(Flux.Optimise.batchmemaybe(d)...)
            end
            d = nothing
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

"""
    registerPSFs(stack, ref_im)

Find the shift between each PSF in `stack` and the reference PSF in `ref_im`
and return the aligned PSFs and their shifts.
 
If `ref_im` has size `(Ny, Nx)`/`(Ny, Nx, Nz)`, `stack` should have size
 `(Ny, Nx, nrPSFs)`/`(Ny, Nx, Nz, nrPSFs)`.
"""
function registerPSFs(stack::Array{T,N}, ref_im) where {T,N}
    @assert N in [3, 4] "stack needs to be a 3d/4d array but was $(N)d"
    ND = ndims(stack)
    Ns = Array{Int,1}(undef, ND - 1)
    Ns .= size(stack)[1:(end - 1)]
    ps = Ns # Relative centers of all correlations
    M = size(stack)[end]
    pad_function = x -> padND(x, ND - 1)

    function crossCorr(
        x::Array{ComplexF64}, y::Array{ComplexF64}, iplan::AbstractFFTs.ScaledPlan
    )
        return fftshift(iplan * (x .* y))
    end

    function norm(x)
        return sqrt(sum(abs2.(x)))
    end

    yi_reg = Array{Float64,N}(undef, size(stack))
    stack_dct = copy(stack)
    ref_norm = norm(ref_im) # norm of ref_im

    # Normalize the stack
    norms = map(norm, eachslice(stack_dct; dims=ND))
    norms = reshape(norms, ones(Int, ND - 1)..., length(norms))
    stack_dct ./= norms
    ref_im ./= ref_norm

    si = zeros(Int, (ND - 1, M))
    # Do FFT registration
    good_count = 1
    dummy_for_plan = Array{eltype(stack_dct),ND - 1}(undef, (2 .* Ns)...)
    plan = plan_rfft(dummy_for_plan; flags=FFTW.MEASURE)
    dummy_for_iplan = Array{ComplexF64,ND - 1}(
        undef, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...
    )
    iplan = plan_irfft(dummy_for_iplan, size(dummy_for_plan)[1]; flags=FFTW.MEASURE)
    pre_comp_ref_im = conj.(plan * (pad_function(ref_im)))
    im_reg = Array{Float64,ND - 1}(undef, Ns...)
    ft_stack = Array{ComplexF64,ND - 1}(undef, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...)
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
        circshift!(im_reg, selectdim(stack, ND, m), si[:, good_count])
        selectdim(yi_reg, ND, good_count) .= im_reg
        good_count += 1
    end
    return collect(selectdim(yi_reg, ND, 1:(good_count - 1))), si
end
