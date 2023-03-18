export make_model
export train_model
export save_model
export load_model

"""    load_model(path; load_optimizer=true, on_gpu=true)

Load a `MultiWienerNet` from a checkpoint saved at `path`. 

Optionally load the optimizer used for training with `load_optimizer`. Returns `(model [, optimizer])`. 

Optionally prepare the network for moving to the GPU if available by setting `on_gpu` (`true` by default).
"""
function load_model(path; load_optimizer=true, on_gpu=true)
    Core.eval(Main, :(using Flux: Flux))
    Core.eval(Main, :(using CUDA: CUDA))
    Core.eval(Main, :(using NNlib: NNlib))
    Core.eval(Main, :(using FFTW: FFTW))
    Core.eval(Main, :(using AbstractFFTs: AbstractFFTs))
    Core.eval(Main, :(using Random: Random))
    if load_optimizer
        @load path model opt
    else
        @load path model
    end
    if model isa Flux.Chain && model[1] isa MultiWienerNet.MultiWiener
        model = Chain(
            MultiWienerNet.toMultiWienerWithPlan(model[1]; on_gpu=on_gpu), model[2]
        )
    end
    if load_optimizer
        return model, opt
    else
        return model
    end
end

"""    make_model(psfs, model_settings::Dict{Symbol, Any}; on_gpu=true)

Create a MultiDeconvolution model and initialize the deconvolution layer with `psfs`.
Entries in `model_settings` affect the UNet unless stated otherwise. The `key=>typeof(value)` pairs in `model_settings` are:
- `:attention => Bool` : use attention gates in skip connection
- `:dropout => Bool` : use dropout layers
- `:depth => Int` : depth of the UNet
- `:separable => Bool` : use separable convolutions in UNet
- `:final_convolution => Bool` : `cat` all activations in the decoder path of the UNet  and pass them through a `tanh` and a convolution before outputting
- `:multiscale => Bool` : Use expensive multiscale convolutions for up- / downscaling
- `:deconv => String` : Which type of deconvolution layer to use. Currently available: `"wiener"`, `"rl"`, `"rl_flfm"`
"""
function make_model(psfs, model_settings::Dict{Symbol,Any}; on_gpu=true)
    # Define Neural Network
    nrPSFs = size(psfs)[end]
    if model_settings[:deconv] == "wiener"
        deconv_stage = MultiWienerNet.MultiWienerWithPlan(psfs; on_gpu=on_gpu)
    elseif model_settings[:deconv] == "rl"
        deconv_stage = RLLayer.RL(psfs)
    elseif model_settings[:deconv] == "rl_flfm"
        deconv_stage = RLLayer_FLFM.RL_FLFM(psfs)
    end
    dimension = ndims(deconv_stage.PSF) + 1
    if size(deconv_stage.PSF, 3) == 1 && deconv_stage isa RLLayer_FLFM.RL_FLFM
        dimension -= 1
    end
    modelUNet = UNet.Unet(
        nrPSFs,
        1,
        dimension;
        up="nearest",
        activation="relu",
        residual=true,
        norm="none",
        model_settings...,
    )
    model = Flux.Chain(deconv_stage, modelUNet)
    return model
end

"""    save_model(model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=nothing)

Save neural network `model` in `checkpointdirectory`.

`losses_train` is the history of losses on training data, which will be read at index `epoch` to get the loss that will be put in the filename.

The model will be saved as `checkpointdirectory/yyyy-mm-ddThh_mm_ss_loss_(losses_train[epoch])_epoch-(epoch+epoch_offset).bson`.

Optionally, the optimizer `opt` used for training can be saved as well, if training should be continued after loading the checkpoint. 
This is only required for stateful opimizers.
"""
function save_model(
    model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=nothing
)
    _ensure_existence(checkpointdirectory)
    model = cpu(model)
    # This check and conversion is necessary because saving a RFFT plan and loading it again results in Julia segfaulting
    if model isa Flux.Chain && model[1] isa MultiWienerNet.MultiWienerWithPlan
        model = Chain(MultiWienerNet.to_multiwiener(model[1]), model[2])
    end # The other types of deconvolution layers don't need special conversion yet
    datestring = replace(string(round(now(), Dates.Second)), ":" => "_")
    modelname =
        datestring *
        "_loss-" *
        string(round(losses_train[epoch]; digits=3)) *
        "_epoch-" *
        string(epoch + epoch_offset) *
        ".bson"
    modelpath = joinpath(checkpointdirectory, modelname)
    if isnothing(opt)
        @save modelpath model
    else
        @save modelpath model opt
    end
    return modelpath
end

"""    setup_training(model, train_x, train_y, validation_x, validation_y, settings)

Helper function to prepare training of the model.

Selects sample data for plotting, prepares the dataset and initializes some arrays storing losses.

Returns one example data point from `validation_x` for plotting, the trainable parameters of `model`, the `DataLoader` containing `train_x` and `train_y`, and two zero-vectors of length `epochs`.
"""
function setup_training(model, train_x, train_y, validation_x, validation_y, settings)
    example_data_x = copy(selectdim(validation_x, ndims(validation_x), 1))
    example_data_x = my_gpu(reshape(example_data_x, size(example_data_x)..., 1))
    example_data_y = copy(selectdim(validation_y, ndims(validation_y), 1))
    example_data_y = reshape(example_data_y, size(example_data_y)..., 1)
    plot_prediction(example_data_y, model[1].PSF, -1, 0, settings.training[:plot_dir])
    pars = Flux.params(model)
    training_datapoints = Flux.DataLoader((train_x, train_y); batchsize=1, shuffle=false)
    epochs = settings.training[:epochs]
    losses_validation = zeros(Float64, epochs)
    losses_train = zeros(Float64, epochs)
    return example_data_x, pars, training_datapoints, losses_validation, losses_train
end

"""    train_model(model, train_x, train_y, validation_x, validation_y, loss, settings; plotloss=false)

Train Flux model `model` on the dataset `(train_x, train_y)`. Each epoch, evaluate performance
on both the training dataset and the validation dataset `(validation_x, validation_y)`, using the loss function `loss`.

Use the `settings` object for all other training related options. For these settings, refer to the documentation.

If `plotloss==true`, plot a graph showing the history of training and validation losses.
"""
function train_model(
    model, train_x, train_y, validation_x, validation_y, loss, settings; plotloss=false
)
    train_data_iterator = Flux.DataLoader((train_x, train_y); batchsize=1, shuffle=true)
    validation_data_iterator = Flux.DataLoader((validation_x, validation_y); batchsize=1)
    example_data_x, pars, training_datapoints, losses_validation, losses_train = setup_training(
        model, train_x, train_y, validation_x, validation_y, settings
    )
    epochs = settings.training[:epochs]
    epoch_offset = settings.checkpoints[:epoch_offset]
    plotevery = settings.training[:plot_interval]
    optimizer = settings.training[:optimizer]
    saveevery = settings.checkpoints[:save_interval]
    early_stopping_patience = settings.training[:early_stopping]
    batchsize = settings.training[:batchsize]
    early_stopping_counter = 0
    for epoch in 1:(epochs - epoch_offset)
        println("Epoch " * string(epoch + epoch_offset) * "/" * string(epochs))
        trainmode!(model, true)
        train_real_gradient!(
            loss, pars, training_datapoints, optimizer; batch_size=batchsize
        )
        trainmode!(model, false)
        losses_train[epoch] = mean(_help_evaluate_loss(train_data_iterator, loss))
        losses_validation[epoch] = mean(_help_evaluate_loss(validation_data_iterator, loss))
        println(
            "\r Loss (train): " *
            string(losses_train[epoch]) *
            ", Loss (validation): " *
            string(losses_validation[epoch]),
        )

        if (saveevery != 0 && epoch % saveevery == 0)
            save_model(
                model,
                settings.checkpoints[:checkpoint_dir],
                losses_train,
                epoch,
                epoch_offset;
                opt=optimizer,
            )
        end

        if (plotevery != 0 && epoch % plotevery == 0)
            pred_to_plot = model(example_data_x)
            psf_to_plot = model[1].PSF
            plot_prediction(
                pred_to_plot, psf_to_plot, epoch, epoch_offset, settings.training[:plot_dir]
            )
            if plotloss
                plot_losses(
                    losses_train, losses_validation, epoch, settings.training[:plot_dir]
                )
            end
        end
        write_to_logfile(
            settings.training[:logfile],
            epoch + epoch_offset,
            losses_train[epoch],
            losses_validation[epoch],
        )
        # Early stopping logic
        if early_stopping_patience > 0 && epoch > 1
            # Chech that new loss is minimal loss and that loss is actually changing
            if losses_validation[epoch] == minimum(losses_validation[1:epoch]) &&
                !iszero(losses_validation[epoch] - losses_validation[epoch - 1])
                savedir = joinpath(settings.checkpoints[:checkpoint_dir], "early_stop")
                # Right now, existence of the early stopping checkpoint folder is not checked while reading the YYAML file -> TODO
                _ensure_existence(savedir)
                foreach(rm, filter(endswith(".bson"), readdir(savedir; join=true)))
                save_model(model, savedir, losses_train, epoch, epoch_offset; opt=optimizer)
                early_stopping_counter = 0
            else
                early_stopping_counter += 1
                @info "Patience reduced to $(early_stopping_patience - early_stopping_counter)"
            end
            if early_stopping_counter >= early_stopping_patience
                @info "Early stopping triggered. Stopping training..."
                break
            end
        end
    end
    # At the end of training, save a checkpoint
    return save_model(
        model,
        settings.checkpoints[:checkpoint_dir],
        losses_train,
        epochs - epoch_offset,
        epoch_offset;
        opt=optimizer,
    )
end
