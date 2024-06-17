using Plots; gr(); Plots.theme(:default)
using Images, FileIO

root = "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/gif2/"
fl  = filter( contains("png"), readdir(root) )
anim = @animate for i in 1:97
    plot(load(joinpath(root, fl[i])), showaxis=false, grid=false, xticks=:none, yticks=:none, size=(1200,443), dpi=300)
end
gif(anim, "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/stitched_black.mp4", fps = 4)



root = "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/gif/"
fl  = filter( contains("png"), readdir(root) )
anim = @animate for i in 1:10
    plot(load(joinpath(root, fl[i])), showaxis=false, grid=false, xticks=:none, yticks=:none, size=(1500,1000), dpi=400)
end
gif(anim, "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/preds_slow.mp4", fps = 2)