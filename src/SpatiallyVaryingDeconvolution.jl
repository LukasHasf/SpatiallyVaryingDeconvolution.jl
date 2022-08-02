module SpatiallyVaryingDeconvolution

export start_training, loadmodel

using YAML
using Images, Colors
using Tullio
using BSON: @save, @load
using Flux
using CUDA
using Statistics
using Dates
using Plots
using KernelAbstractions, CUDAKernels
using ProgressMeter
include("UNet.jl")
include("MultiWienerNet.jl")
include("utils.jl")
include("losses.jl")
include("plotting.jl")
include("model.jl")

function start_training(; T=Float32, kwargs...)
    options = Dict(kwargs)
    # Load and process the data
    psfs = readPSFs(options[:psfs_path], options[:psfs_key])
    if options[:center_psfs]
        options[:psf_ref_index] = if options[:psf_ref_index] == -1
            size(psfs)[end] ÷ 2 + 1
        else
            options[:psf_ref_index]
        end
        psfs, _ = registerPSFs(
            psfs, collect(selectdim(psfs, ndims(psfs), options[:psf_ref_index]))
        )
    end
    x_data, y_data = load_data(
        options[:nrsamples],
        options[:truth_dir],
        options[:sim_dir];
        newsize=options[:newsize],
    )
    x_data = applynoise(x_data)
    x_data = x_data .* convert(eltype(x_data), 2) .- one(eltype(x_data))
    y_data = y_data .* convert(eltype(y_data), 2) .- one(eltype(y_data))
    train_x, test_x = train_test_split(x_data)
    train_y, test_y = train_test_split(y_data)

    # Define / load the model
    dims = length(options[:newsize])
    optimizer = options[:optimizer]
    if !options[:load_checkpoints]
        nrPSFs = size(psfs)[end]
        resized_psfs = Array{T,dims + 1}(undef, options[:newsize]..., nrPSFs)
        for i in 1:nrPSFs
            selectdim(resized_psfs, dims + 1, i) .= imresize(
                collect(selectdim(psfs, dims + 1, i)), options[:newsize]
            )
        end
        resized_psfs = my_gpu(resized_psfs)
        model = my_gpu(
            makemodel(
                resized_psfs;
                depth=options[:depth],
                attention=options[:attention],
                dropout=options[:dropout],
                separable=options[:separable],
                final_attention=options[:final_attention],
            ),
        )
    else
        model, optimizer = my_gpu(loadmodel(options[:loadpath]))
    end
    pretty_summarysize(x) = Base.format_bytes(Base.summarysize(x))
    println("Model takes $(pretty_summarysize(cpu(model))) of memory.")
    # Define the loss function
    kernel = _get_default_kernel(dims; T=T)

    loss_fn = let model = model, kernel = kernel
        function loss_fn(x, y)
            return L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    end

    # Training
    return train_model(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_fn;
        epochs=options[:epochs],
        epoch_offset=options[:epoch_offset],
        checkpointdirectory=options[:checkpoint_dir],
        plotloss=true,
        plotevery=options[:plot_interval],
        optimizer=optimizer,
        plotdirectory=options[:plot_dir],
        saveevery=options[:save_interval],
        logfile=options[:logfile],
    )
end

function start_training(options_path; T=Float32)
    options = read_yaml(options_path)
    return start_training(; T=T, options...)
end

end # module
