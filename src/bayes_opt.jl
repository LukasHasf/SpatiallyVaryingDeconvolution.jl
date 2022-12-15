using SpatiallyVaryingDeconvolution
using Flux
using LaplaceRedux
Random.seed!(54321)

function start_training_ml(options_path; T=Float32)
    # First, train a neural network
    model_path = start_training_ml(options_path; T=T)
    # Load trained model
    nn = load_model(model_path; load_optimizer=false, on_gpu=false)
    la = Laplace(nn; likelihood=:regression)
    train_x, train_y, test_x, test_y = prepare_data(settings; T=T)
    data = zip(train_x, train_y)
    fit!(la, data)
    optimize_prior!(la)
    # Then use la(X) for inference. Will return mean and variance
end