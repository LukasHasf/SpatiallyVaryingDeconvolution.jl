# Examples

This directory contains some files that show example usage of this package.

- [`options.yaml`](options.yaml) is a configuration file for a 2D deconvolution training
- [`options3D.yaml`](options3D.yaml) is a configuration file for 3D training
- [`applyOnDir.jl`](applyOnDir.jl) provides functions to let the model run on every observation in a directory and save the deconvolved observations to another directory
- [`incremental_train.jl`](incremental_train.jl) provides a method to use a model trained at one resolution for training a model of higher resolution, as a sort of transfer learning method.