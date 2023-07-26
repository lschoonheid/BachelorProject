# Kaggle CSV file constants
DETECTOR_KEYS = ["volume_id", "layer_id", "module_id"]
HITS_SAMPLES = None
TABLE_INDEX = {
    "hits": 0,
    "cells": 1,
    "particles": 2,
    "truth": 3,
}

# TODO: move this to dirs
# Local file constants
DIRECTORY = "/data/atlas/users/lschoonh/BachelorProject/"
CACHE_LOC = "/data/atlas/users/lschoonh/BachelorProject/.pickle_cache/"
DATA_ROOT = DIRECTORY + "data/"
DATA_SAMPLE = DATA_ROOT + "train_100_events/"

# Matplotlib constants
FIG_X = 10
FIG_Y = FIG_X
FIG_DPI = 600
PLOT_FONTSIZE = 6
FIG_EXTENSION = ".jpg"
