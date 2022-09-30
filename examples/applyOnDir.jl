using SpatiallyVaryingDeconvolution
using Images
using Tullio
include("../src/utils.jl")

"""    deblur(source_dir, destination_dir, checkpoint_path)

Run model saved at `checkpoint_path` on a directory `source_dir` of PNG images and
save the resulting images in `destination_dir`.
"""
function deblur(source_dir, destination_dir, checkpoint_path)
    model = load_model(checkpoint_path; load_optimizer=false)
    for filepath in readdir(source_dir; join=true)
        if !endswith(filepath, ".png") 
            continue
        end
        # Model was trained using Float32 data
        file = Float32.(Gray.(load(filepath))) 
        # Model expects same size it was trained with
        file = imresize(file, (64,  64))
        # Model expects input of shape [size_y, size_x, nr_channels, batch_size]
        file = reshape(file, size(file)..., 1, 1)
        # Model was trained to expect data in range [-1, 1]
        file = _map_to_zero_one(file) .* 2 .- 1
        deblurred = model(file)
        # Image saving expects data to be in range [0, 1]
        deblurred = _map_to_zero_one(deblurred)
        savepath = joinpath(destination_dir, splitpath(filepath)[end])
        save(savepath, deblurred[:, :, 1, 1])
    end
end