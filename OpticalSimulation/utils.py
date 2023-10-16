import numpy as np
from scipy import fftpack
import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append('..')
import Basics.sensorParams as psp

# import matplotlib.pyplot as plt
def fast_poisson(gx, gy):
    #    j = 1:ydim-1;
    #	k = 1:xdim-1;
    #
    #	% Laplacian
    #	gyy(j+1,k) = gy(j+1,k) - gy(j,k);
    #	gxx(j,k+1) = gx(j,k+1) - gx(j,k);

    m, n = gx.shape
    gxx = np.zeros((m, n))
    gyy = np.zeros((m, n))
    f = np.zeros((m, n))
    img = np.zeros((m, n))
    gyy[1:, :-1] = gy[1:, :-1] - gy[:-1, :-1]
    gxx[:-1, 1:] = gx[:-1, 1:] - gx[:-1, :-1]
    f = gxx + gyy

    f2 = f[1:-1, 1:-1].copy()

    f_sinx = fftpack.dst(f2, norm='ortho')
    f_sinxy = fftpack.dst(f_sinx.T, norm='ortho').T

    x_mesh, y_mesh = np.meshgrid(range(n - 2), range(m - 2))
    x_mesh = x_mesh + 1
    y_mesh = y_mesh + 1
    denom = (2 * np.cos(np.pi * x_mesh / (n - 1)) - 2) + (2 * np.cos(np.pi * y_mesh / (m - 1)) - 2)

    f3 = f_sinxy / denom
    #    plt.figure(10)
    #    plt.imshow(denom)
    #    plt.show()
    # inverse discrete sine transform
    f_realx = fftpack.idst(f3, norm='ortho')
    f_realxy = fftpack.idst(f_realx.T, norm='ortho').T
    img[1:-1, 1:-1] = f_realxy.copy()
    return img

def getDomeHeightMap(heightMap:np.ndarray, press_depth:float, domeMap:np.ndarray):
    """_summary_

    Args:
        heightMap (np.ndarray): mm
        press_depth (float): mm
        domeMap (np.ndarray): dome map model (image)

    Returns:
        _type_: height map with contact, contact_mask
    """
    heightMap /= psp.pixmm
    max_o = np.max(heightMap)
    heightMap -= max_o
    pressing_height_pix = press_depth/psp.pixmm

    gel_map = heightMap+pressing_height_pix

    contact_mask = (gel_map > domeMap)
    zq = np.zeros((psp.d,psp.d))
    zq[contact_mask] = gel_map[contact_mask] - domeMap[contact_mask]
    return zq, contact_mask
