# Example configuration file
This module expects all the configuration to be in a single YAML file, which should look like this:

```yaml
data:
    # Path to blurred data
    x_path : ../training_data/Data/Simulated_Miniscope_2D_Training_data/
    # Path to ground truth
    y_path : ../training_data/Data/Ground_truth_downsampled/
    # Data needs to be resized to be a power of two along each axis
    resize_to : [64, 64]
    # Should the data be centered?
    center_psfs: true
    # If center_psfs, which index corresponds to the central PSF? -1 means length \div 2 + 1
    reference_index: -1

model:
    # Depth of UNet
    depth: 3
    # Use attention gates when combining activations of encoding and decoding branch
    attention: true
    # Use dropout layers in convolution blocks
    dropout: true
    # Use separable convolution in convolution blocks
    separable: true
    # Concatenate all UNet activations and concolve them as a final step
    final_attention: true

training:
    # Path to the PSF file
    psfs_path: ../SpatiallyVaryingConvolution/comaPSF.mat
    # Key under which the PSFs are stored
    psfs_key: psfs
    # How many training+testing samples to load from x_path and y_path
    nrsamples: 1000
    # How many epochs to train
    epochs: 50
    # The employed optimizer
    optimizer: ADADelta
    # How often should a plot of the evaluation of the model on the first test data be plotted? 0 for never
    plot_interval: 1
    # Where should that plot be saved?
    plot_path: examples/training_progress/
    # Should the losses be written to a log file?
    log_losses: false
    checkpoints:
        # Should a previously trained model be loaded? [true, false, latest]
        load_checkpoints: latest
        # If load_checkpoints, what is the path to the checkpoint?
        checkpoint_path: nothing
        # Directory where new checkpoints will be saved
        checkpoint_dir: examples/checkpoints/
        # How often should a checkpoint be saved? 0 for never except for at the end of training
        save_interval: 1
```

## The options in detail
Here's a list of what each field in the configuration field does:
- `x_path`, `y_path` : The paths to the ground_truth and training dat directories. See [Preparing the training data](index.md#preparing-the-training-data)
- `resize_to` : The UNet implementation requires samples to be powers of 2 in size along each spatial dimension. Each sample is resized to the dimension specified by this.
- `center_psfs`, `reference_index` : If your PSFs are not already centered, set `center_psfs` to `true` and set the `reference_index` to the PSF closest to the center.
- `depth` : The number of downsampling / upsampling steps in the UNet.
- `attention` : Boolean to indicate if the UNet should use attention gates.
- `dropout` : Boolean to indicate if the UNet should employ dopout-layers during training.
- `separable` : Whether to use separable or regular convolutions in the UNet convolution layers.
- `final_attention` : Whether to add a convolution layer which processes all intermediate (upsampled) activations in the decoder path followed by an attention gate.
- `psfs_path`, `psfs_key` : Path to file containing the PSFs. `mat` files have `dict`-like structure, so you also need to provide the key with which one can access the PSFs array.
- `nrsamples` : The number of samples to load and train with. They will be divided into 70% training and 30% testing data.
- `epochs` : The number of epochs the model will be trained.
- `log_losses` : Boolean to indicate if the train and test loss after each epoch should be saved into a file.
- `plot_interval`, `plot_path` : Plot the result of using the model on the first testing sample every `plot_interval`-th epoch and save the result in the directory `plot_path`. Set `plot_interval` to `0` to deactivate.
- `load_checkpoints`, `checkpoint_path` : You can continue training from a previously saved checkpoint. If you want to do so, set `load_checkpoints` to `true` and provide the path to the checkpoint you want to load. Alternatively, set `load_checkpoints` to `latest` to load the most recent checkpoint in `checkpoint_dir`. In that case, `checkpoint_path` is ignored.
- `checkpoint_dir`, `save_interval` : During training, every `save_interval`-th epoch, a checkpoint will be saved into the directory `checkpoint_dir`. Set `save_interval` to `0` to disable this. At the end of training, a checkpoint will be saved regardless.