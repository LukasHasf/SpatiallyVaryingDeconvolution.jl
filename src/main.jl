"""    start_training(; T=Float32, kwargs...)

Start the training of a deconvolution network.

All numeric data will be of type `T` (default `Float32`).

If this function signature is called, all of the following keywords
    need to be included in `kwargs`:

- `psfs_path::String` : Path to the file containing the psfs

- `psfs_key::String` : Key to access the PSFs in `psfs_path`

- `center_psfs::Bool` : Indicates if PSFs should be centered

- `psf_ref_index<:Integer` : Index of the reference PSF in the PSFs array

- `nrsamples<:Integer` : Number of samples to load for training and testing

- `truth_dir::String` : Path to directory containing ground truth samples

- `sim_dir::String` : Path to directory containing simulated samples

- `newsize::Tuple{Int}` : Size of resized sample data.

- `optimizer<:Flux.Optimise.AbstractOptimiser` : An instance of a `Flux` optimizer used for training the network

- `load_checkpoints::Bool` : Indicates whether a checkpoint should be loaded instead of starting from scratch

- `depth<:Integer` : Depth of the UNet

- `attention::Bool` : Indicates whether to use attention gates in the UNet

- `dropout::Bool` : Indicates whether to use dropout layers in the conv-blocks of the UNet

- `separable::Bool` : Indicates whether to use separable convolution filters in a conv-block

- `final_attention::Bool` : Indicates whether to use a final layer appending all activations in the expanding path and attention gating them

- `checkpoint_path::String` : (If `load_checkpoints`) Path of checkpoint to load

- `epochs<:Int` : Number of epochs to train

- `checkpoint_dir::String` : Directory in which to store checkpoints 

- `plot_interval<:Integer` : Plot prediction of model on a test sample every `plot_interval` epochs

- `plot_dir::String` : Directory in which to save the generated plots

- `save_interval<:Integer` : Save a checkpoint every `save_interval` epochs

- `logfile::Union{nothing, String}` : Write to a logfile located at `logfile` ( or don't, if `isnothing(logfile)`)
"""
function start_training(settings::Settings; T=Float32)
    # Load and process the data
    psfs = readPSFs(settings.training[:psfs_path], settings.training[:psfs_key])
    psfs = _center_psfs(psfs, settings.data[:center_psfs], settings.data[:psf_ref_index])
    x_data, y_data = load_data(
        settings.training[:nrsamples],
        settings.data[:truth_dir],
        settings.data[:sim_dir];
        newsize=settings.data[:newsize],
    )
    x_data = apply_noise(x_data)
    x_data = x_data .* convert(eltype(x_data), 2) .- one(eltype(x_data))
    y_data = y_data .* convert(eltype(y_data), 2) .- one(eltype(y_data))
    train_x, test_x = train_test_split(x_data)
    train_y, test_y = train_test_split(y_data)

    # Define / load the model
    dims = length(settings.data[:newsize])
    optimizer = settings.training[:optimizer]
    if !settings.checkpoints[:load_checkpoints]
        nrPSFs = size(psfs)[end]
        resized_psfs = Array{T,dims + 1}(undef, settings.data[:newsize]..., nrPSFs)
        for i in 1:nrPSFs
            selectdim(resized_psfs, dims + 1, i) .= imresize(
                collect(selectdim(psfs, dims + 1, i)), settings.data[:newsize]
            )
        end
        resized_psfs = my_gpu(resized_psfs)
        model = my_gpu(make_model(resized_psfs, settings.model))
    else
        model, optimizer = my_gpu(load_model(settings.checkpoints[:checkpoint_path]))
        settings.training[:optimizer] = optimizer
    end
    println("Model takes $(pretty_summarysize(cpu(model))) of memory.")
    # Define the loss function
    kernel = _get_default_kernel(dims; T=T)
    kernel = my_gpu(reshape(kernel, size(kernel)..., 1, 1))

    loss_fn = let model = model, kernel = kernel
        function loss_fn(x, y)
            return L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    end
    display(settings.checkpoints)
    # Training
    return train_model(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_fn;
        plotloss=true,
        epochs=settings.training[:epochs],
        epoch_offset=settings.checkpoints[:epoch_offset],
        checkpointdirectory=settings.checkpoints[:checkpoint_dir],
        plotevery=settings.training[:plot_interval],
        optimizer=settings.training[:optimizer],
        plotdirectory=settings.training[:plot_dir],
        saveevery=settings.checkpoints[:save_interval],
        logfile=settings.training[:logfile],
    )
end

"""    start_training(options_path; T=Float32)

Start the training of a deconvolution network.

`options_path` is the path to the configuration `YAML` file.
All numeric data will be of type `T` (default `Float32`).
"""
function start_training(options_path; T=Float32)
    options = Settings(options_path)
    return start_training(options; T=T)
end
