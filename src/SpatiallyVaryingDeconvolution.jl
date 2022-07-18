module SpatiallyVaryingDeconvolution

export start_training

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

function nn_convolve(img::AbstractArray{T,N}; kernel=AbstractArray{T}) where {T,N}
    @assert ndims(img) == ndims(kernel) + 2
    kernel = my_gpu(reshape(kernel, size(kernel)..., 1, 1))
    convolved = conv(my_gpu(img), kernel;)
    return convolved
end

function SSIM_loss(
    ŷ::AbstractArray{T,N}, y::AbstractArray{T,N}; kernel=nothing
) where {T,N}
    c1 = T(0.01)^2
    c2 = T(0.03)^2
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
    return one(T) - mean(ssim_map)
end

function L1_loss(ŷ, y)
    return Flux.Losses.mae(y, ŷ)
end

function L1_SSIM_loss(ŷ, y; kernel=nothing)
    return L1_loss(ŷ, y) + SSIM_loss(ŷ, y; kernel=kernel)
end

function sliced_plot(arr)
    l = @layout [a b; c d]
    clim = extrema(arr)
    p_yx = heatmap(
        arr[:, :, end ÷ 2 + 1]; clim=clim, colorbar=false, ylabel="y", ticks=false
    )
    p_yz = heatmap(
        arr[:, end ÷ 2 + 1, :]; clim=clim, colorbar=false, xlabel="z", ticks=false
    )
    p_xz = heatmap(
        arr[end ÷ 2 + 1, :, :]';
        clim=clim,
        colorbar=false,
        ylabel="z",
        xlabel="x",
        ticks=false,
    )

    my_colorbar = scatter(
        [0, 0],
        [1, 0];
        zcolor=[0, 3],
        clims=clim,
        xlims=(1, 1.1),
        framstyle=:none,
        label="",
        grid=false,
        xshowaxis=false,
        yshowaxis=false,
        ticks=false,
    )

    return plot(p_yx, p_yz, p_xz, my_colorbar; layout=l)
end

function plot_prediction(prediction, psf, epoch, epoch_offset, plotdirectory)
    prediction = cpu(prediction)
    psf = cpu(psf)
    if ndims(psf) == 3
        # 2D case -> prediction is (Ny, Nx, channels, batchsize)
        p1 = heatmap(prediction[:, :, 1, 1])
        p2 = heatmap(abs2.(psf[:, :, 1]))
    elseif ndims(psf) == 4
        # 3D case -> prediction is (Ny, Nx, Nz, channels, batchsize)
        p1 = sliced_plot(prediction[:, :, :, 1, 1])
        p2 = sliced_plot(abs2.(psf[:, :, :, 1]))
    end
    prediction_path = joinpath(
        plotdirectory, "Epoch" * string(epoch + epoch_offset) * "_predict.png"
    )
    psf_path = joinpath(
        plotdirectory, "LearnedPSF_epoch" * string(epoch + epoch_offset) * ".png"
    )
    savefig(p1, prediction_path)
    return savefig(p2, psf_path)
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

function start_training(options_path; T=Float32)
    options = read_yaml(options_path)
    # Load and process the data
    psfs = readPSFs(options["psfs path"], options["psfs key"])
    if options["center psfs"]
        options["psf ref index"] = if options["psf ref index"] == -1
            size(psfs)[end] ÷ 2 + 1
        else
            options["psf ref index"]
        end
        psfs, _ = registerPSFs(
            psfs, collect(selectdim(psfs, ndims(psfs), options["psf ref index"]))
        )
    end
    x_data, y_data = load_data(
        options["nrsamples"],
        options["truth dir"],
        options["sim dir"];
        newsize=options["newsize"],
    )
    x_data = applynoise(x_data)
    train_x, test_x = train_test_split(x_data)
    train_y, test_y = train_test_split(y_data)

    # Define / load the model
    dims = length(options["newsize"])
    optimizer = options["optimizer"]
    if !options["load checkpoints"]
        nrPSFs = size(psfs)[end]
        resized_psfs = Array{T,dims + 1}(undef, options["newsize"]..., nrPSFs)
        for i in 1:nrPSFs
            selectdim(resized_psfs, dims + 1, i) .= imresize(
                collect(selectdim(psfs, dims + 1, i)), options["newsize"]
            )
        end
        resized_psfs = my_gpu(resized_psfs)
        model = my_gpu(
            makemodel(
                resized_psfs;
                depth=options["depth"],
                attention=options["attention"],
                dropout=options["dropout"],
                separable=options["separable"],
                final_attention=options["final attention",],
            ),
        )
    else
        model, optimizer = my_gpu(loadmodel(options["loadpath"]))
    end
    pretty_summarysize(x) = Base.format_bytes(Base.summarysize(x))
    println("Model takes $(pretty_summarysize(cpu(model))) of memory.")
    # Define the loss function
    kernel = T.(_get_default_kernel(dims))

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
        epochs=options["epochs"],
        epoch_offset=options["epoch offset"],
        checkpointdirectory=options["checkpoint dir"],
        plotloss=true,
        plotevery=options["plot interval"],
        optimizer=optimizer,
        plotdirectory=options["plot dir"],
        saveevery=options["save interval"],
        logfile=options["logfile"],
    )
end

end # module
