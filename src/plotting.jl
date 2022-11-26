export plot_prediction, plot_losses

function sliced_plot(arr)
    l = @layout [a b; c d]
    clim = extrema(arr)
    p_yx = heatmap(
        maximum(arr; dims=3)[:, :, 1]; clim=clim, colorbar=false, ylabel="y", ticks=false, c=:grays
    )
    p_yz = heatmap(
        maximum(arr; dims=2)[:, 1, :]; clim=clim, colorbar=false, xlabel="z", ticks=false, c=:grays
    )
    p_xz = heatmap(
        maximum(arr; dims=1)[1, :, :]';
        clim=clim,
        colorbar=false,
        ylabel="z",
        xlabel="x",
        ticks=false, 
        c=:grays,
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
        c=:grays,
    )

    return plot(p_yx, p_yz, p_xz, my_colorbar; layout=l)
end

function plot_prediction(prediction, psf, epoch, epoch_offset, plotdirectory)
    prediction = cpu(prediction)
    psf = cpu(psf)
    if ndims(psf) == 3
        # 2D case -> prediction is (Ny, Nx, channels, batchsize)
        p1 = heatmap(prediction[:, :, 1, 1]; c=:grays)
        p2 = heatmap(abs2.(psf[:, :, 1]); c=:grays)
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
    plot(train_loss[1:epoch]; label="Train loss")
    xlabel!("Epochs")
    ylabel!("Loss")
    plot!(test_loss[1:epoch]; label="Test loss")
    return savefig(joinpath(plotdirectory, "lossplot.png"))
end
