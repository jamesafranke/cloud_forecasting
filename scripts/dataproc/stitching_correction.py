import os, random, datetime, numpy as np, pandas as pd, pylab as plt
from glob import glob
from zipfile import ZipFile
from satpy.scene import Scene
import datashader as das
import dask.array as da
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

root = '/share/data/2pals/jim/data/'


