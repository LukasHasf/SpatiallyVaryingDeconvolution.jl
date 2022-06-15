module SpatiallyVaryingDeconvolution

export start_training

using YAML
using Images, Colors
using Tullio
using BSON: @save, @load
using Flux
using Statistics
using Dates
using Plots
include("UNet.jl")
include("MultiWienerNet.jl")
include("utils.jl")

function loadmodel(path)
    @load path model
    return model
end

function makemodel(psfs)
    # Define Neural Network
    nrPSFs = size(psfs)[end]
    modelwiener = MultiWienerNet.MultiWiener(psfs) #|> gpu
    modelUNet = UNet.Unet(
        nrPSFs,
        1,
        ndims(psfs) + 1;
        up="nearest",
        activation="relu",
        residual=true,
        norm="None",
        attention=true,
    )
    model = Flux.Chain(modelwiener, modelUNet)
    return model
end

function nn_convolve(img::Array{T,N}; kernel=nothing) where {T,N}
    kernel = T.(kernel)
    if ndims(kernel) == 2
        if N == 3
            img = view(img, :, :, 1)
        elseif N == 4
            @tullio convolved[x + _, y + _, a, b] := img[x + i, y + j, a, b] * kernel[i, j]
            return convolved
        end
    elseif ndims(kernel) == 3
        # TODO: This is likely wrong
        if N == 5
            img = view(img, :, :, :, 1, 1)
            @tullio convolved[x + _, y + _, z + _] :=
                img[x + i, y + j, z + k] * kernel[i, j, k]
            return convolved
        elseif N == 3
            @tullio convolved[x + _, y + _, z + _] :=
                img[x + i, y + j, z + k] * kernel[i, j, k]
            return convolved
        end
    end
end

function SSIM_loss(ŷ, y; kernel=nothing)
    c1 = (0.01)^2
    c2 = (0.03)^2
    mu1 = nn_convolve(y; kernel=kernel)
    mu2 = nn_convolve(ŷ; kernel=kernel)
    mu1_sq = mu1 .^ 2
    mu2_sq = mu2 .^ 2
    mu1_mu2 = mu1 .* mu2
    sigma1 = nn_convolve(y .^ 2; kernel=kernel) .- mu1_sq
    sigma2 = nn_convolve(ŷ .^ 2; kernel=kernel) .- mu2_sq
    sigma12 = nn_convolve(y .* ŷ; kernel=kernel) .- mu1_mu2
    ssim_map = @. ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
        ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return one(eltype(y)) - oftype(y[1], mean(ssim_map))
end

function L1_loss(ŷ, y)
    return Flux.Losses.mae(y, ŷ)
end

function L1_SSIM_loss(ŷ, y; kernel=nothing)
    return L1_loss(ŷ, y) + SSIM_loss(ŷ, y; kernel=kernel)
end

function plot_prediction(prediction, psf, epoch, epoch_offset, plotdirectory)
    if ndims(psf) == 3
        # 2D case -> prediction is (Ny, Nx, channels, batchsize)
        heatmap(prediction[:, :, 1, 1])
        savefig(plotdirectory * "Epoch" * string(epoch + epoch_offset) * "_predict.png")
        heatmap(abs2.(psf[:, :, 1]))
        savefig(plotdirectory * "LearnedPSF_epoch" * string(epoch + epoch_offset) * ".png")
    end
end

function plot_losses(train_loss, test_loss, epoch, plotdirectory)
    plot(train_loss[1:epoch])
    xlabel!("Epochs")
    ylabel!("Loss")
    savefig(joinpath(plotdirectory, "trainlossplot.png"))
    plot(test_loss[1:epoch])
    xlabel!("Epochs")
    ylabel!("Loss")
    return savefig(joinpath(plotdirectory, "testlossplot.png"))
end

function train_real_gradient!(loss, ps, data, opt)
    # Zygote calculates a complex gradient, even though this is mapping  real -> real.
    # Might have to do with fft and incomplete Wirtinger derivatives? Anyway, only
    # use the real part of the gradient
    for (i, d) in enumerate(data)
        try
            gs = Flux.gradient(ps) do
                loss(Flux.Optimise.batchmemaybe(d)...)
            end
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

function saveModel(model, checkpointdirectory, losses_train, epoch, epoch_offset)
    datestring = string(round(now(), Dates.Second))
    modelname =
        datestring *
        "_loss-" *
        string(round(losses_train[epoch]; digits=3)) *
        "_epoch-" *
        string(epoch + epoch_offset) *
        ".bson"
    modelpath = joinpath(checkpointdirectory, modelname)
    @save modelpath model
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
    optimizer=Flux.Optimise.ADAM,
)
    example_data_x = collect(selectdim(test_x, ndims(test_x), 1))
    example_data_x = reshape(example_data_x, size(example_data_x)..., 1)
    example_data_y = collect(selectdim(test_y, ndims(test_y), 1))
    example_data_y = reshape(example_data_y, size(example_data_y)[1:2]...)
    pars = Flux.params(model)
    training_datapoints = Flux.Data.DataLoader(
        (train_x, train_y); batchsize=1, shuffle=false
    )
    opt = optimizer()
    losses_test = zeros(Float64, epochs)
    losses_train = zeros(Float64, epochs)
    for epoch in 1:(epochs - epoch_offset)
        println("Epoch " * string(epoch + epoch_offset) * "/" * string(epochs))
        train_real_gradient!(loss, pars, training_datapoints, opt)
        losses_train[epoch] = loss(train_x, train_y)
        losses_test[epoch] = loss(test_x, test_y)
        print(
            "\r Loss (train): " *
            string(losses_train[epoch]) *
            ", Loss (test): " *
            string(losses_test[epoch]),
        )

        if saveevery > 0 && epoch % saveevery == 0
            saveModel(model, checkpointdirectory, losses_train, epoch, epoch_offset)
        end

        if plotevery > 0 && epoch % plotevery == 0
            pred_to_plot = model(example_data_x)
            psf_to_plot = model[1].PSF
            plot_prediction(pred_to_plot, psf_to_plot, epoch, epoch_offset, plotdirectory)
            if plotloss
                plot_losses(losses_train, losses_test, epoch, plotdirectory)
            end
        end
        print("\n")
    end
    # At the end of training, save a checkpoint
    return saveModel(
        model, checkpointdirectory, losses_train, epochs - epoch_offset, epoch_offset
    )
end

function start_training(options_path; T=Float32)
    # Define dictionaries
    optimizer_dict = Dict(
        "ADAM" => Flux.Optimise.ADAM,
        "Descent" => Flux.Optimise.Descent,
        "ADAMW" => Flux.Optimise.ADAMW,
        "ADAGrad" => Flux.Optimise.ADAGrad,
        "ADADelta" => Flux.Optimise.ADADelta,
    )

    # Load options
    options = YAML.load_file(options_path)
    optimizer_kw = options["training"]["optimizer"]
    @assert optimizer_kw in keys(optimizer_dict) "Optimizer $optimizer_kw not defined"
    truth_directory = options["data"]["x_path"]
    simulated_directory = options["data"]["y_path"]
    newsize = tuple(options["data"]["resize_to"]...)
    loadpath = nothing
    if options["training"]["checkpoints"]["load_checkpoints"]
        loadpath = options["training"]["checkpoints"]["checkpoint_path"]
        # TODO: Find epoch_offset based on checkpoint name
    end
    nrsamples = options["training"]["nrsamples"]
    epochs = options["training"]["epochs"]
    plotevery = options["training"]["plot_interval"]
    plotpath = options["training"]["plot_path"]
    plotpath = endswith(plotpath, "/") ? plotpath : plotpath * "/"
    if !isdir(plotpath)
        mkpath(plotpath)
    end
    center_psfs = options["data"]["center_psfs"]
    if center_psfs
        psf_ref_index = options["data"]["reference_index"]
    end
    checkpoint_dir = options["training"]["checkpoints"]["checkpoint_dir"]
    if !isdir(checkpoint_dir)
        mkpath(checkpoint_dir)
    end
    saveevery = options["training"]["checkpoints"]["save_interval"]
    optimizer = optimizer_dict[optimizer_kw]

    # Load and process the data
    psfs = readPSFs(options["training"]["psfs_path"], options["training"]["psfs_key"])
    if center_psfs
        psf_ref_index = psf_ref_index == -1 ? size(psfs)[end] ÷ 2 + 1 : psf_ref_index
        psfs, _ = registerPSFs(psfs, collect(selectdim(psfs, ndims(psfs), psf_ref_index)))
    end
    x_data, y_data = load_data(
        nrsamples, truth_directory, simulated_directory; newsize=newsize
    )
    x_data = applynoise(x_data)
    train_x, test_x = train_test_split(x_data)
    train_y, test_y = train_test_split(y_data)

    # Define / load the model
    dims = length(newsize)
    if isnothing(loadpath)
        nrPSFs = size(psfs)[end]
        resized_psfs = Array{T,dims + 1}(undef, newsize..., nrPSFs)
        for i in 1:nrPSFs
            selectdim(resized_psfs, dims + 1, i) .= imresize(
                collect(selectdim(psfs, dims + 1, i)), newsize
            )
        end
        model = makemodel(resized_psfs)
    else
        model = loadmodel(loadpath)
    end

    # Define the loss function
    if dims == 3
        @tullio kernel[x, y, z] :=
            gaussian(11, 1.5)[x] * gaussian(11, 1.5)[y] * gaussian(11, 1.5)[z]
    elseif dims == 2
        kernel = gaussian(11, 1.5) .* gaussian(11, 1.5)'
    end

    loss_fn = let model = model, kernel = kernel
        function loss_fn(x, y)
            return L1_SSIM_loss(model(x), y; kernel=kernel)
        end
    end

    # Test so far
    selection_x = collect(selectdim(train_x, ndims(train_x), 1:10))
    selection_y = collect(selectdim(train_x, ndims(train_x), 1:10))
    reshaped_size = size(train_x)[1:(end - 1)]
    display(
        loss_fn(
            reshape(selection_x, reshaped_size..., 10),
            reshape(selection_y, reshaped_size..., 10),
        ),
    )

    # Training
    return train_model(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        loss_fn;
        epochs=epochs,
        checkpointdirectory=checkpoint_dir,
        plotloss=true,
        plotevery=plotevery,
        optimizer=optimizer,
        plotdirectory=plotpath,
        saveevery=saveevery,
    )
end

end # module
