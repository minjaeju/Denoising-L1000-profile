from __future__ import unicode_literals, print_function, division
from pdb import set_trace

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import os

    
def show_plot(train_points, plot_every, fold, eval_points=None, save_path='./result', file_name='train', save_as_img=True):
    # plt.figure()
    ax = plt.figure().gca()
    # fig, ax = plt.subplots()
    plt.title('%s %s fold loss' % (file_name, str(fold)))

    # train loss
    x1 = list(range(1, len(train_points)*plot_every+1, plot_every))
    plt.plot(x1, train_points, label='train_loss')
    
    # valid loss
    if eval_points is not None:
        x2 = list(range(1, len(eval_points)*plot_every+1, plot_every))
        plt.plot(x2, eval_points, label='valid_loss')

    # add legend 
    plt.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_as_img:
        plt.savefig(os.path.join(save_path, '%s_fold%d.png' % (file_name, fold)))

        
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

    
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
