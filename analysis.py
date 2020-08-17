# coding: utf-8
# Echo Chamber Model w/ Social Capital Incentives
# analysis.py
# Last Update: 20200816
# by Kazutoshi Sasahara
# modified by Erik WEis

import pandas as pd
import numpy as np
import scipy.stats as stats
import peakutils


def screen_diversity(content_values, bins):
    h, w = np.histogram(content_values, range=(-1, 1), bins=bins)
    return stats.entropy(h+1, base=2)


def num_opinion_peaks(opinions):
    nparam_density = stats.kde.gaussian_kde(opinions)
    x = np.linspace(-1, 1, 100)
    density = nparam_density(x)
    indexes = peakutils.indexes(density, thres=0, min_dist=10)
    print(x[indexes], density[indexes])
    return len(indexes)