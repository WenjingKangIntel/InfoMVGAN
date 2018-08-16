import numpy as np
import math
import os
import scipy.signal
import scipy.ndimage


mb_size = 64

def psnr(img_t, img_g):
    count = 0
    for idx in range(mb_size):
        mse = np.mean(np.square((img_t[idx] - img_g[idx])))
        # print (mse)
        if mse == 0:
            psresult = 100
        else:
            PIXEL_MAX = 255.0
            psresult = 10 * np.log10(np.square(PIXEL_MAX) / mse)
        count += psresult
    avgcount = count / (mb_size)
    return avgcount


def nqi(X_t, X_g):
    count = 0
    for idx in range(mb_size):
        mean_x_t = np.mean(X_t[idx,:,:,0])
        mean_x_g = np.mean(X_g[idx,:,:,0])
        var_x_t = (4096.0 / 4095.0) * np.var(X_t[idx,:,:,0])
        var_x_g = (4096.0 / 4095.0) * np.var(X_g[idx,:,:,0])
        computecount = 0
        for i in range(64):
            for j in range(64):
                computecount += (X_t[idx][i][j][0] - mean_x_t) * (X_g[idx][i][j][0] - mean_x_g)
        covar = computecount / (4095.0)
        result = (4.0*covar*mean_x_t*mean_x_g)/(math.pow(mean_x_g, 2.0)+math.pow(mean_x_t, 2.0))/(var_x_t+var_x_t)
        count += result
    avgcount = count / (mb_size)
    return avgcount

def ssim(X_t, X_g):
    count = 0
    for idx in range(mb_size):
        mean_x_t = np.mean(X_t[idx, :, :, 0])
        mean_x_g = np.mean(X_g[idx, :, :, 0])
        var_x_t = (4096.0 / 4095.0) * np.var(X_t[idx, :, :, 0])
        var_x_g = (4096.0 / 4095.0) * np.var(X_g[idx, :, :, 0])

        computecount = 0
        for i in range(64):
            for j in range(64):
                computecount += (X_t[idx][i][j][0] - mean_x_t) * (X_g[idx][i][j][0] - mean_x_g)
        covar = computecount / (4095.0)

        result = (2.0 * mean_x_t * mean_x_g + 0.0001) * (2.0 * covar + 0.0009) / (
                math.pow(mean_x_t, 2.0) + math.pow(mean_x_g, 2.0) + 0.0001) / (var_x_t + var_x_g + 0.0009)
        count += result
    avgcount = count / (mb_size)
    return avgcount

def vifp(ref, dist):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp
