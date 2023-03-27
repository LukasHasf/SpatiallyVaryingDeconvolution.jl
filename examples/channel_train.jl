using SpatiallyVaryingDeconvolution
include("../src/utils.jl")
using YAML
using Flux

function channel_train(options_path)
    settings = Settings(options_path)
    train_x, train_y, test_x, test_y = prepare_data(settings; T=Float32)
    dims = length(settings.data[:newsize])
    prepare_model!(settings)
    println("Model takes $(pretty_summarysize(cpu(model))) of memory.")
    # Define the loss function
    kernel = _get_default_kernel(dims; T=T)
    kernel = my_gpu(reshape(kernel, size(kernel)..., 1, 1))
    psfs = prepare_psfs(settings)

    settings.data[:channels] = 1
    models = []

    for c in axes(train_x, dims-1)
        train_x_c = selectdim(train_x, dims-1, c)
        test_x_c = selectdim(test_x, dims-1, c)
        train_y_c = selectdim(train_y, dims-1, c)
        test_y_c = selectdim(test_y, dims-1, c)
        psfs_c = selectdim(psfs, ndims(psfs)-1, c)
        model = make_model(psfs_c, settings.model)
        loss_fn = let model = model, kernel = kernel
            function loss_fn(x, y)
                return L1_SSIM_loss(model(x), y; kernel=kernel)
            end
        end
        checkpoint_path = train_model(model, train_x_c, train_y_c, test_x_c, test_y_c, loss_fn, settings)
        push!(models, checkpoint_path)
    end
    print(models)

    models = [load_model(model_path; load_optimizer=false, on_gpu=false) for model_path in models]
    settings = Settings(options_path)
    _, _, test_x, test_y = prepare_data(settings; T=Float32)
    losses = zeros(size(test_x, ndims(test_x)))
    mkpath("channelwise/prediction/")
    mkpath("channelwise/gt/")
    mkpath("channelwise/forward/")
    for i in axes(test_x, ndims(test_x))
        inp = selectdim(test_x, ndims(test_x), i)
        gt = selectdim(test_y, ndims(test_y), i)
        inp = reshape(inp, size(inp)..., 1)
        gt = reshape(gt, size(gt)..., 1)
        pred = cat([m(selectdim(inp, ndims(inp)-1, i)) for (i,m) in enumerate(models)]...; dims=ndims(inp)-1)
        losses[i] = Flux.Losses.mae(gt, pred)
        pred_img = _rgb_to_img(pred)
        save("channelwise/prediction/$i.png", pred_img)
        save("channelwise/gt/$i.png", _rgb_to_img(selectdim(gt, ndims(gt), 1)))
        save("channelwise/forward/$i.png", _rgb_to_img(selectdim(inp, ndims(inp), 1)))
    end
    # Write lossses to file
    outfile = "channelwise_loss.csv"
    open(outfile, "w") do f
        for l in losses
            println(f, l)
        end
    end
end