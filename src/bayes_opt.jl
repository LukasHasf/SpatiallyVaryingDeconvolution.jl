using SpatiallyVaryingDeconvolution
using Flux
using Zygote
using Turing
using Distributions
Turing.setadbackend(:zygote)

struct Likelihood
    network
    sigma
end
Flux.@functor Likelihood
(p::Likelihood)(x) = Normal.(p.network(x)[:], p.sigma[1])

@model function TuringModel(likelihood_conditional, weight_prior, sigma_prior, X, y)
    weights ~ weight_prior
    sigma ~ sigma_prior
    nn = likelihood_conditional(weights, sigma)
    predictions = [nn(reshape(x, size(x)...,1)) for x in eachslice(X; dims=ndims(X))]
    println(size(predictions))
    println(size(predictions[1]))
    println(size(y))
    for i in axes(y, ndims(y))
        y[:, :, 1, i][:] .~ predictions[i]
    end
end

function start_training_ml(options_path; T=Float32)
    # First, train a neural network
   # model_path = start_training(options_path; T=T)
    model_path = "examples/checkpoints/2022-12-15T16_35_31_loss-0.0_epoch-100.bson"
    println("Training done. Model saved at $model_path.")
    # Load trained model
    nn = load_model(model_path; load_optimizer=false, on_gpu=false)
    settings = Settings(options_path)
    train_x, train_y, test_x, test_y = prepare_data(settings; T=T)
    #println("Passing data through loaded network: ", nn(reshape(train_x[:, :, :, 1], size(train_x)[1:3]...,1)))
    data = zip(train_x, train_y)
    likelihood = Likelihood(nn, ones(1,1))
    θ, re = Flux.destructure(likelihood)
    n_weights = length(θ) - 1
    println("$n_weights weights to optimize.")
    likelihood_conditional(weights, sigma) = re(vcat(weights..., sigma))
    # Define prior
    weight_prior = MvNormal(zeros(n_weights), ones(n_weights))
    sigma_prior = Gamma(1.,1.)
    tmodel = TuringModel(likelihood_conditional, weight_prior, sigma_prior, train_x, train_y)
    N = 10
    ch = sample(tmodel, HMC(0.025, 4), N)
    # Get posterior
    weights = MCMCChains.group(ch, :weights).value
    sigmas = MCMCChains.groudp(ch, :sigma).value
end