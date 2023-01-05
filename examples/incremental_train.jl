using SpatiallyVaryingDeconvolution
using YAML
include("../src/utils.jl")
include("../src/MultiWienerNet.jl")
include("../src/RLLayer.jl")
using Images

using Flux

function transfer_train(old_model_path, new_resolution::Tuple, options_path)
    old_model = load_model(old_model_path; load_optimizer=false)
    unet = old_model[2]
    deconv_layer = old_model[1]
    old_learned_psfs = deconv_layer.PSF
    new_PSF = imresize(old_learned_psfs, new_resolution)
    new_deconv = deconv_layer
    if deconv_layer isa MultiWienerNet.MultiWiener
        old_learned_λ = old_model[1].lambda
        new_deconv = MultiWienerNet.MultiWiener(new_PSF, old_learned_λ)
        new_deconv = MultiWienerNet.toMultiWienerWithPlan(new_deconv)
    elseif deconv_layer isa RLLayer.RL
        old_n_iter = deconv_layer.n_iter
        new_deconv = RLLayer.RL(new_PSF, old_n_iter)
    end
    new_model = Flux.Chain(new_deconv, unet)
    options = Settings(options_path)
    options.data[:newsize] = new_resolution
    new_model = my_gpu(new_model)
    return start_training(new_model, options)
end

#=
function transfer_train(old_model_path, new_resolution :< Integer, options_path)
    options = read_yaml(options_path)
    new_res = options[:newsize] .* new_resolution
    return transfer_train(old_model_path, new_res, options_path)
end =#

function start_training(model, settings::Settings; T=Float32)
    # Load and preprocess the data
    train_x, train_y, test_x, test_y = prepare_data(settings; T=T)
    # Set right optimizer
    prepare_model!(settings)

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
        model, train_x, train_y, test_x, test_y, loss_fn, settings; plotloss=true
    )
end
