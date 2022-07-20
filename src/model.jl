export makemodel
export train_model
export saveModel
export loadmodel

function loadmodel(path; load_optimizer=true)
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

function makemodel(
    psfs; attention=true, dropout=true, depth=3, separable=false, final_attention=true
)
    # Define Neural Network
    nrPSFs = size(psfs)[end]
    psfs = psfs
    modelwiener = MultiWienerNet.MultiWienerWithPlan(psfs)
    modelUNet = UNet.Unet(
        nrPSFs,
        1,
        ndims(psfs) + 1;
        up="nearest",
        activation="relu",
        residual=true,
        norm="none",
        attention=attention,
        depth=depth,
        dropout=dropout,
        separable=separable,
        final_attention=final_attention,
    )
    model = Flux.Chain(modelwiener, modelUNet)
    return model
end

function saveModel(
    model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=nothing
)
    model = cpu(model)
    if model isa Flux.Chain
        model = Chain(MultiWienerNet.toMultiWiener(model[1]), model[2])
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

function train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    loss;
    epochs=1,
    epoch_offset=0,
    plotloss=false,
    plotevery=10,
    plotdirectory="training_progress/",
    saveevery=1,
    checkpointdirectory="checkpoints/",
    optimizer=Flux.Optimise.ADAM(),
    logfile=nothing,
)
    example_data_x = copy(selectdim(test_x, ndims(test_x), 1))
    example_data_x = my_gpu(reshape(example_data_x, size(example_data_x)..., 1))
    example_data_y = copy(selectdim(test_y, ndims(test_y), 1))
    example_data_y = reshape(example_data_y, size(example_data_y)..., 1)
    pars = Flux.params(model)
    training_datapoints = Flux.Data.DataLoader(
        (train_x, train_y); batchsize=1, shuffle=false
    )
    losses_test = zeros(Float64, epochs)
    losses_train = zeros(Float64, epochs)
    for epoch in 1:(epochs - epoch_offset)
        println("Epoch " * string(epoch + epoch_offset) * "/" * string(epochs))
        trainmode!(model, true)
        train_real_gradient!(loss, pars, training_datapoints, optimizer)
        trainmode!(model, false)
        losses_train[epoch] = mean([
            _help_evaluate_loss(train_x, train_y, i, loss) for
            i in 1:size(train_x, ndims(train_x))
        ])
        losses_test[epoch] = mean([
            _help_evaluate_loss(test_x, test_y, i, loss) for
            i in 1:size(test_x, ndims(test_x))
        ])
        print(
            "\r Loss (train): " *
            string(losses_train[epoch]) *
            ", Loss (test): " *
            string(losses_test[epoch]),
        )

        if saveevery > 0 && epoch % saveevery == 0
            saveModel(
                model, checkpointdirectory, losses_train, epoch, epoch_offset; opt=optimizer
            )
        end

        if plotevery > 0 && epoch % plotevery == 0
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
    return saveModel(
        model,
        checkpointdirectory,
        losses_train,
        epochs - epoch_offset,
        epoch_offset;
        opt=optimizer,
    )
end
