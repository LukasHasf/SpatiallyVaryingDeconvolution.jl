# SpatiallyVaryingDeconvolution.jl

This package provides model definitions and training facilities for the MultiWienerNet described in [[1]](#Sources). This model is capable of reverting the effect of a spatially varying convolution in 2D and 3D.

|  **Build Status**                        |  **Code Coverage**  |
|:----------------------------------------:|:-------------------:|
| [![][CI-img]][CI-url]                    |[![][CC-img]][CC-url]|

## Installation
```julia
julia> ] add https://github.com/LukasHasf/SpatiallyVaryingDeconvolution.jl
```

## Quickstart
Training the model is as simple as calling `train_model` on your configuration file:
```julia
using SpatiallyVaryingDeconvolution
start_training("myOptions.yaml")
```

`myOptions.yaml` should use all the fields the [`options.yaml`](examples/options.yaml) in [examples](examples) defines.

## Preparing the training data

In order to train the model, you need to provide training data, ground truth data and the PSFs. Per sample of data, there should be a file in the training data directory and the ground truth data directory each. Supported file formats are `png` for 2D and `mat` or HDF5 for 3D . An example directory structure might look like this:
```
├── ground_truth_data
│   ├── 001.png
│   ├── 002.png
│   └── 003.png
└── training_data
    ├── 001.png
    ├── 002.png
    └── 003.png
```
The PSFs should be provided in a single Matlab/HDF5 file as an 3D/4D array with the 2/3 first dimension being the spatial dimensions.

The paths to the ground truth data directory, the training data directory and the PSFs should be set in your `options.yaml` in the fields `x_path`, `y_path` and `psfs_path`, respectively. Note that if you use relative paths, they will be relative to your current working directory and not the directory where the `options.yaml` file is located.

## Other options
The [`options.yaml`](examples/options.yaml) provides a few more configuration options, so here is a list of what each field does:
- `x_path`, `y_path` : The paths to the ground_truth and training dat directories. See [Preparing the training data](#preparing-the-training-data)
- `resize_to` : The UNet implementation requires samples to be powers of 2 in size along each spatial dimension. Each sample is resized to the dimension specified by this.
- `center_psfs`, `reference_index` : If your PSFs are not already centered, set `center_psfs` to `true` and set the `reference_index` to the PSF closest to the center.
- `SNR` : Signal to noise ratio of the training data. Noise is applied to the training data. The strength of the noise is chosen such that the noisy training data has a SNR close to the one given in the options.
- `depth` : The number of downsampling / upsampling steps in the UNet.
- `attention` : Boolean to indicate if the UNet should use attention gates.
- `dropout` : Boolean to indicate if the UNet should employ dopout-layers during training.
- `separable` : Whether to use separable or regular convolutions in the UNet convolution layers.
- `final_convolution` : Whether to add a convolution layer which processes all intermediate (upsampled) activations in the decoder path after a `tanh` activation function.
- `multiscale` : Whether to use multiscale convolutions instead of normal ones. Increases performance in conjuction with transfer training, but requires more memory (significantly more in 3D).
- `deconv` :  Which type of deconvolution layer to use. Currently available: `"wiener"`, `"rl"`, `"rl_flfm"`. `"wiener"` for a Wiener deconvolution layer, `"rl"` for Richardson-Lucy deconvolution layer and `"rl_flfm"` for a Richardson-Lucy deconvolution that is adapted to Fourier Light Field Microscopy (2D observation -> 3D reconstruction).
- `psfs_path`, `psfs_key` : Path to file containing the PSFs. `mat` files have `dict`-like structure, so you also need to provide the key with which one can access the PSFs array.
- `nrsamples` : The number of samples to load and train with. They will be divided into 70% training and 30% validation data.
- `epochs` : The number of epochs the model will be trained.
- `early_stopping` : Stop training if validation loss doesn't decrease within the given number of epochs and reset weights to the best performing model concerning validation loss.
- `weight_decay`: Apply weight decay regularization.
- `log_losses` : Boolean to indicate if the train and validation loss after each epoch should be saved into a file.
- `plot_interval`, `plot_path` : Plot the result of using the model on the first validation sample every `plot_interval`-th epoch and save the result in the directory `plot_path`. Set `plot_interval` to `0` to deactivate.
- `load_checkpoints`, `checkpoint_path` : You can continue training from a previously saved checkpoint. If you want to do so, set `load_checkpoints` to `true` and provide the path to the checkpoint you want to load. Alternatively, set `load_checkpoints` to `latest` to load the most recent checkpoint in `checkpoint_dir`.
- `checkpoint_dir`, `save_interval` : During training, every `save_interval`-th epoch, a checkpoint will be saved into the directory `checkpoint_dir`. Set `save_interval` to `0` to disable this. At the end of training, a checkpoint will be saved regardless.

## Usage after training
After training, you will have a fully trained MultiWienerNet saved as a checkpoint. This checkpoint can be loaded and used like a normal function. Due to some problems with BSON, a few packages need to be loaded in the main session:
```julia
using SpatiallyVaryingDeconvolution
using Flux
using CUDA
using NNlib
using FFTW
using AbstractFFTs
using Random
model = load_model(checkpoint_path; load_optimizer=false)
# Apply model to new blurry data
deblurred = model(blurry)
```


## Sources
[1] : Yanny, K., Antipa, N., Liberti, W., Dehaeck, S., Monakhova, K., Liu, F. L., Shen, K., Ng, R., & Waller, L. (2020). Miniscope3D: optimized single-shot miniature 3D fluorescence microscopy. In Light: Science &amp; Applications (Vol. 9, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1038/s41377-020-00403-7 

[CI-img]: https://github.com/LukasHasf/SpatiallyVaryingDeconvolution.jl/workflows/CI/badge.svg
[CI-url]: https://github.com/LukasHasf/SpatiallyVaryingDeconvolution.jl/actions?query=workflow%3ACI 
[CC-img]: https://codecov.io/gh/LukasHasf/SpatiallyVaryingDeconvolution.jl/branch/master/graph/badge.svg?token=9Q5HNIVNV8
[CC-url]: https://codecov.io/gh/LukasHasf/SpatiallyVaryingDeconvolution.jl
