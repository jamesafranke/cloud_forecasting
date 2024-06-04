using Plots; gr(); Plots.theme(:default)
using Images, FileIO

root = "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/gif/"
fl  = filter( contains("png"), readdir(root) )
anim = @animate for i in 1:131
    plot(load(joinpath(root, fl[i])), showaxis=false, grid=false, xticks=:none, yticks=:none, size=(600,600), dpi=300)
end
gif(anim, "/Users/jamesfranke/Documents/julia/cloud_forecasting/figs/goesband4710.mp4", fps = 4)