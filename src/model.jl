export make_model
export train_model
export save_model
export load_model

"""    load_model(path; load_optimizer=true)

Load a `MultiWienerNet` from a checkpoint saved at `path`. Optionally load the optimizer 
used for training with `load_optimizer`. Returns `(model [, optimizer])`
"""
function load_model(path; load_optimizer=true)
    Core.eval(Main, :(using Flux: Flux))
    Core.eval(Main, :(using CUDA: CUDA))
    Core.eval(Main, :(using NNlib: NNlib))
    Core.eval(Main, :(using FFTW: FFTW))
    Core.eval(Main, :(using AbstractFFTs: AbstractFFTs))
    if load_optimizer
        @load path model opt
        model = Chain(MultiWienerNet.toMultiWienerWithPlan(model[1]), model[2])
        return model, opt
    else
        @load path model
        if model isa Flux.Chain
            model = Chain(MultiWienerNet.toMultiWienerWithPlan(model[1]), model[2])
        end
        return model
    end
end

"""    make_model(psfs; kwargs)

Create a MultiWiener model and initialize the Wiener deconvolution with `psfs`.
Keyword arguments are:
- `attention::Bool` : use attention gates in skip connection
- `dropout::Bool` : use dropout layers
- `depth::Int` : depth of the UNet
- `separable::Bool` : use separable convolutions in UNet
- `final_attention::Bool` : `cat` all activations in the decoder path of the UNet
 and pass them through an attention gate and a convolution before outputting
"""
function make_model(
    psfs, model_settings::Dict{Symbol, Any}
)
    # Define Neural Network
    nrPSFs = size(psfs)[end]
    modelwiener = MultiWienerNet.MultiWienerWithPlan(psfs)
    modelUNet = UNet.Unet(
        nrPSFs,
        1,
        ndims(psfs) + 1;
        up="nearest",
        activation="relu",
        residual=true,
        norm="none",
        model_settings...
    )
    model = Flux.Chain(modelwiener, modelUNet)
    return model
end

function save_model(
    model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=nothing
)
    model = cpu(model)
    if model isa Flux.Chain
        model = Chain(MultiWienerNet.to_multiwiener(model[1]), model[2])
    end
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

function setup_training(model, train_x, train_y, test_x, test_y, settings::Settings)
    example_data_x = copy(selectdim(test_x, ndims(test_x), 1))
    example_data_x = my_gpu(reshape(example_data_x, size(example_data_x)..., 1))
    example_data_y = copy(selectdim(test_y, ndims(test_y), 1))
    example_data_y = reshape(example_data_y, size(example_data_y)..., 1)
    plot_prediction(example_data_y, model[1].PSF, -1, 0, settings.training[:plot_dir])
    pars = Flux.params(model)
    training_datapoints = Flux.Data.DataLoader(
        (train_x, train_y); batchsize=1, shuffle=false
    )
    epochs = settings.training[:epochs]
    losses_test = zeros(Float64, epochs)
    losses_train = zeros(Float64, epochs)
    return example_data_x, pars, training_datapoints, losses_test, losses_train
end
    for epoch in 1:(epochs - epoch_offset)
        println("Epoch " * string(epoch + epoch_offset) * "/" * string(epochs))
        trainmode!(model, true)
        train_real_gradient!(loss, pars, training_datapoints, optimizer; batch_size=1)
        trainmode!(model, false)
        losses_train[epoch] = mean(
            _help_evaluate_loss(Flux.DataLoader((train_x, train_y); batchsize=1), loss)
        )
        losses_test[epoch] = mean(
            _help_evaluate_loss(Flux.DataLoader((test_x, test_y); batchsize=1), loss)
        )
        print(
            "\r Loss (train): " *
            string(losses_train[epoch]) *
            ", Loss (test): " *
            string(losses_test[epoch]),
        )

        if (saveevery != 0 && epoch % saveevery == 0)
            save_model(
                model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=optimizer
            )
        end

        if (plotevery != 0 && epoch % plotevery == 0)
            pred_to_plot = model(example_data_x)
            psf_to_plot = model[1].PSF
            plot_prediction(pred_to_plot, psf_to_plot, epoch, epoch_offset, plotdirectory)
            if plotloss
                plot_losses(losses_train, losses_test, epoch, plotdirectory)
            end
        end
        write_to_logfile(
            logfile, epoch + epoch_offset, losses_train[epoch], losses_test[epoch]
        )
        print("\n")
    end
    # At the end of training, save a checkpoint
    return save_model(
        model,
        checkpointdirectory,
        losses_train,
        epochs - epoch_offset,
        epoch_offset;
        opt=optimizer,
    )
end
