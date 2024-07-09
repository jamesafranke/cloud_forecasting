using Plots; gr(); Plots.theme(:default)
using Images, FileIO

root = "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/cloudbench_gif/"
fl  = sort(filter( contains("jpg"), readdir(root) ))
anim = @animate for i in 1:400
    plot(load(joinpath(root, fl[i])), showaxis=false, grid=false, xticks=:none, yticks=:none, size=(1800,700), dpi=300)
end
gif(anim, "/Users/jamesfranke/Documents/julia/cloud_forecasting/cloudbench.mp4", fps = 5)


root = "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/gif/"
fl  = filter( contains("png"), readdir(root) )
anim = @animate for i in 1:10
    plot(load(joinpath(root, fl[i])), showaxis=false, grid=false, xticks=:none, yticks=:none, size=(1500,1000), dpi=400)
end
gif(anim, "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/preds_slow.mp4", fps = 2)
