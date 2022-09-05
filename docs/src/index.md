# SpatiallyVaryingDeconvolution.jl

This package provides model definitions and training facilities for the MultiWienerNet described in [[1]](#Sources). This model is capable of reverting the effect of a spatially varying convolution in 2D and 3D.

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

`myOptions.yaml` should use all the fields the [example configuration file](example_conf.md) defines.

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

## Usage after training
After training, you will have a fully trained MultiWienerNet saved as a checkpoint. This checkpoint can be loaded and used like a normal function:
```julia
using SpatiallyVaryingDeconvolution
model = load_model(checkpoint_path; load_optimizer=false)
# Apply model to new blurry data
deblurred = model(blurry)
```


## Sources
[1] : Yanny, K., Antipa, N., Liberti, W., Dehaeck, S., Monakhova, K., Liu, F. L., Shen, K., Ng, R., & Waller, L. (2020). Miniscope3D: optimized single-shot miniature 3D fluorescence microscopy. In Light: Science &amp; Applications (Vol. 9, Issue 1). Springer Science and Business Media LLC. https://doi.org/10.1038/s41377-020-00403-7 