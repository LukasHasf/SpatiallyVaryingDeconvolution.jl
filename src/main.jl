"""    start_training(settings::Settings; T=Float32)

Start the training of a deconvolution network.

All numeric data will be of type `T` (default `Float32`).

# Arguments
- `psfs_path::String` : Path to the file containing the psfs

- `psfs_key::String` : Key to access the PSFs in `psfs_path`

- `center_psfs::Bool` : Indicates if PSFs should be centered

- `psf_ref_index<:Integer` : Index of the reference PSF in the PSFs array

- `nrsamples<:Integer` : Number of samples to load for training and validation

- `truth_dir::String` : Path to directory containing ground truth samples

- `sim_dir::String` : Path to directory containing simulated samples

- `newsize::Tuple{Int}` : Size of resized sample data.

- `optimizer<:Flux.Optimise.AbstractOptimiser` : An instance of a `Flux` optimizer used for training the network

- `load_checkpoints::Bool` : Indicates whether a checkpoint should be loaded instead of starting from scratch

- `depth<:Integer` : Depth of the UNet

- `attention::Bool` : Indicates whether to use attention gates in the UNet

- `dropout::Bool` : Indicates whether to use dropout layers in the conv-blocks of the UNet

- `separable::Bool` : Indicates whether to use separable convolution filters in a conv-block

- `final_convolution::Bool` : Whether to add a layer at the end of the U-NEt which convolves the outputs of the decoder path with a 1x1 kernel after passing a `tanh` activation function.

- `checkpoint_path::String` : (If `load_checkpoints`) Path of checkpoint to load

- `epochs<:Int` : Number of epochs to train

- `checkpoint_dir::String` : Directory in which to store checkpoints 

- `plot_interval<:Integer` : Plot prediction of model on a validation sample every `plot_interval` epochs

- `plot_dir::String` : Directory in which to save the generated plots

- `save_interval<:Integer` : Save a checkpoint every `save_interval` epochs

- `logfile::Union{nothing, String}` : Write to a logfile located at `logfile` ( or don't, if `isnothing(logfile)`)
"""
function start_training(settings::Settings; T=Float32)
    # Load and process the data
    train_x, train_y, validation_x, validation_y = prepare_data(settings; T=T)

    # Define / load the model
    dims = length(settings.data[:newsize])
    model = prepare_model!(settings)
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
        validation_x,
        validation_y,
        loss_fn,
        settings;
        plotloss=true,
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
