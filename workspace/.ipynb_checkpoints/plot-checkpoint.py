# -*- coding: utf-8 -*-
# +
import numpy as np
import matplotlib.pyplot as plt

from bwim import reconstruction


# -

def show_signal(truck, figsize=(9,5)):
    plt.figure(figsize=figsize)
    meters = truck.time * truck.speed
    plt.plot(meters, truck.signals.T, zorder=2)
    for p in truck.peaks:
        plt.axvline(meters[p], linestyle='--', color='k', alpha=0.1)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.title('Speed: ' + str(np.round(truck.speed * 3.6, decimals=2)) + ' km/h')
    plt.xlabel('Meters')
    plt.show()


def compare_weights(estimated, groundthruth):
    index = np.arange(len(estimated)) + 1
    bar_width = 0.45
    ver_shift = 1
    opacity = 0.8
    error = error = np.abs(estimated - groundthruth).sum()
    plt.figure(figsize=(9,5))
    plt.bar(index-bar_width/2, estimated,    bar_width, alpha=opacity, color='b', label='Pesées en marche')
    plt.bar(index+bar_width/2, groundthruth, bar_width, alpha=opacity, color='r', label='Pesées statiques')
    plt.title('Erreur total: {:2.2f} t'.format(error), fontsize=14)
    plt.xlabel('Essieu', fontsize=14)
    plt.ylabel('Poid', fontsize=14)
    plt.legend(fontsize=14)
    for x, y, z in zip(index, groundthruth, estimated):
        plt.text(x+bar_width/2, y-ver_shift, '%.2f' % y, fontsize=12, fontweight='bold', color='white', ha='center')
        plt.text(x-bar_width/2, z-ver_shift, '%.2f' % z, fontsize=12, fontweight='bold', color='white', ha='center')
    plt.show()


def show_calibration(truck, influence):
    reconstructed, rescaled = reconstruction(truck, influence)
    meters = truck.speed * truck.time
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(meters[:len(rescaled)], rescaled)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.xlabel('Meters')
    plt.subplot(1,2,2)
    plt.plot(meters, reconstructed, linewidth=2, label='Recon.')
    plt.plot(meters, truck.signals.T, label='Observed', alpha=0.7)
    plt.axhline(0, color='k', linewidth=1, zorder=0, alpha=0.1)
    plt.xlabel('Meters')
    plt.legend()
    plt.show()
