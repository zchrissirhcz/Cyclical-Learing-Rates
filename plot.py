#!/usr/bin/env python2
# coding: utf-8

"""
This code means to give straight-forward visualization of the proposed CLR policy.
"""

# lr_policy = "triangular"

import matplotlib.pyplot as plt

def rate_triangular(t, s, b, m):
    """
    t: current iter
    s: stepsize
    b: base_lr
    m: max_lr
    """
    c = t / (2*s)  # c: cycle
    x = t - (2*c+1)*s
    x = x*1.0 / s
    r = b + (m-b)*max(0, 1-abs(x))
    return r

def rate_triangular2(t, s, b, m):
    """
    t: current iter
    s: stepsize
    b: base_lr
    m: max_lr
    """
    c = t / (2*s)  # c: cycle
    x = t - (2*c+1)*s
    x = x*1.0 / s
    r = b + (m-b)*min(1, max(0, (1-abs(x))/pow(2.0, c)))
    return r

def plot_lr(lr_rate_fun, title_name, save_name):
    s = 25
    b = 0.01
    m = 0.05
    t_list = []
    r_list = []
    for t in range(200):
        #r = rate_triangular(t, s, b, m)
        r = lr_rate_fun(t, s, b, m)
        t_list.append(t)
        r_list.append(r)
    plt.plot(t_list, r_list)
    plt.xlabel('iteration')
    plt.ylabel('learing rate')
    plt.title(title_name)
    plt.show()
    

if __name__ == '__main__':
    plot_lr(rate_triangular, 'triangular policy', 'lr_triangular.jpg')
    plot_lr(rate_triangular2, 'triangular2 policy', 'lr_triangular2.jpg')
