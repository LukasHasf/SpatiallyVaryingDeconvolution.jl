using SpatiallyVaryingDeconvolution
using Tullio
using YAML
include("../src/utils.jl")
include("../src/MultiWienerNet.jl")
using Images

using Flux

function transfer_train(old_model_path, new_resolution::Tuple, options_path)
    old_model = load_model(old_model_path; load_optimizer=false)
    unet = old_model[2]
    old_learned_psfs = old_model[1].PSF
    old_learned_λ = old_model[1].lambda
    new_PSF = imresize(old_learned_psfs, new_resolution)
    new_multiwiener = MultiWienerNet.MultiWiener(new_PSF, old_learned_λ)
    new_multiwiener = MultiWienerNet.toMultiWienerWithPlan(new_multiwiener)
    new_model = Flux.Chain(new_multiwiener, unet)
    options = read_yaml(options_path)
    options[:newsize] = new_resolution
    new_model = my_gpu(new_model)
    start_training(new_model; options...)
end

#=
function transfer_train(old_model_path, new_resolution :< Integer, options_path)
    options = read_yaml(options_path)
    new_res = options[:newsize] .* new_resolution
    return transfer_train(old_model_path, new_res, options_path)
end =#

function start_training(model; T=Float32, kwargs...)
    options = Dict(kwargs)
    # Load and process the data
    psfs = readPSFs(options[:psfs_path], options[:psfs_key])
    psfs = _center_psfs(psfs, options[:center_psfs], options[:psf_ref_index])
    x_data, y_data = load_data(
        options[:nrsamples],
        options[:truth_dir],
        options[:sim_dir];
        newsize=options[:newsize],
    )
    x_data = apply_noise(x_data)
    x_data = x_data .* convert(eltype(x_data), 2) .- one(eltype(x_data))
    y_data = y_data .* convert(eltype(y_data), 2) .- one(eltype(y_data))
    train_x, test_x = train_test_split(x_data)
    train_y, test_y = train_test_split(y_data)

    # Define / load the model
    dims = length(options[:newsize])
    optimizer = options[:optimizer]
    println("Model takes $(pretty_summarysize(cpu(model))) of memory.")
    # Define the loss function
    kernel = _get_default_kernel(dims; T=T)
    kernel = my_gpu(reshape(kernel, size(kernel)..., 1, 1))

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