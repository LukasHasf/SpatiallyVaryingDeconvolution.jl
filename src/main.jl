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
function start_training(; T=Float32, kwargs...)
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
            make_model(
                resized_psfs;
                depth=options[:depth],
                attention=options[:attention],
                dropout=options[:dropout],
                separable=options[:separable],
                final_attention=options[:final_attention],
            ),
        )
    else
        model, optimizer = my_gpu(load_model(options[:checkpoint_path]))
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

"""    start_training(options_path; T=Float32)

Start the training of a deconvolution network.

`options_path` is the path to the configuration `YAML` file.
All numeric data will be of type `T` (default `Float32`).
"""
function start_training(options_path; T=Float32)
    options = read_yaml(options_path)
    return start_training(; T=T, options...)
end