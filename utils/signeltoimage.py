import torch
from scipy.signal import spectrogram
import numpy as np
from tftb.processing import *
import cv2
import pywt
# from scipy.signal import kaiser, hamming
from scipy.signal import resample
from pyts.image import GramianAngularField
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def resampe(sgn,samplenum=2):
    sgn = resample(sgn, sgn.shape[1] // samplenum, axis=1)
    return sgn

def resampe2(sgn,samplenum=2):
    sgn = np.resize(sgn, (2,sgn.shape[1] // samplenum))
    return sgn

def resampe3(sgn,samplenum=2):
    groups = np.split(sgn, samplenum, axis=1)

    # 对每个分组求均值，得到一个（2，1）的数组
    means = [np.mean(group, axis=1, keepdims=True) for group in groups]

    # 将所有的均值数组沿着第二个维度拼接起来，得到一个（2，1024/samplenum）的数组
    b = np.concatenate(means, axis=1)
    return b

def stp(data,resize_is=False,nperseg=42,lap=40,norm='maxmin',resize_num=128,nfft=128):
    t = np.linspace(0, 1, 10001)
    # fs = 1 / (t[1] - t[0])
    fs = 32e6
    sgn=data
    if sgn.shape[0]==2:
        sgn = data[0, :] + data[1, :] * 1j

    f, tt, p = spectrogram(x=sgn,fs=fs, nperseg=nperseg,
                               noverlap=lap,
                               return_onesided=True,nfft=nfft)

    if norm=='linalg':
        normImg = p/np.linalg.norm(p)
    elif norm=='maxmin':
        normImg = p
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min())
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(p)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'denoise':
        normImg = p / np.linalg.norm(p)
        # normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() )
        Max=np.max(normImg)
        Min=np.min(normImg)
        yuzhi=(9/10)*np.percentile(normImg,25)+(1/10)*(Max+Min)
        normImg =np.where(normImg>yuzhi,normImg ,0)
    if resize_is==True:
        normImg = cv2.resize(normImg, (resize_num,resize_num))
        normImg = np.expand_dims(normImg, axis=0)
    else:
        normImg = np.expand_dims(normImg, axis=0)

    return normImg

def converet_RGB(img,cmap='rainbow'):
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=img.min(), vmax=img.max())
    rgba_data = cmap(norm(img))
    # rgba_data = cmap(img)
    rgba_data = np.delete(rgba_data, 3, axis=2)
    rgb_data = np.transpose(rgba_data, (2, 0, 1))
    return rgb_data

def pwvd(data,norm='no',resize_is=False,resize_num=128,RGB_is=False,cmap='rainbow'):
    t = np.linspace(0, 1, data.shape[1])
    sgn = data[0, :] + data[1, :] * 1j
    spec = PseudoWignerVilleDistribution(sgn, timestamps=t)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    img,_,_=spec.run()

    if resize_is==True:
        img=cv2.resize(img, (resize_num,resize_num))
    if norm=='maxmin':
        # img = np.abs(np.log(1 + np.abs(img)))
        ampLog=img
        img = (ampLog - ampLog.min()) / (ampLog.max() - ampLog.min())

    if norm == 'meanvar':
        img = (img - np.mean(img)) / (np.std(img) + 1.e-9)
    if norm == 'maxmin2':
        Min=-0.000999912436681158
        Max=0.0013195788363637715
        img = (img - Min) / (Max - Min +1e-6)
    if norm == 'maxmin2022':
        Min=-69.02861173465756
        Max=423.1396374466868
        img = (img - Min) / (Max - Min +1e-6)
    if norm == 'meanstd':
        mean=7.318065201053581e-05
        std=0.00025146489151817126
        img=(img-mean)/std
    if RGB_is==True:
        img=converet_RGB(img,cmap=cmap)
    if RGB_is == False:
        img=np.expand_dims(img, axis=0)

    return img

def gasf(data,norm='no',resize_is=False,resize_num=128,RGB_is=False,sample_range=None,cmap='rainbow'):
    gaf = GramianAngularField(method='summation',sample_range=sample_range, overlapping=True)
    # sgn0=np.expand_dims(data,0)
    # sgn1 = np.expand_dims(data[1], 0)
    img0 = gaf.fit_transform(data)
    # img1 = gaf.fit_transform(sgn1)
    img = np.mean(img0,0)
    # img=np.squeeze(img,0)



    if resize_is==True:
        img=cv2.resize(img, (resize_num,resize_num))
    if norm=='maxmin':
        # img = np.abs(np.log(1 + np.abs(img)))
        for i in range(img.shape[0]):
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min()+1e-6)

    if norm == 'meanstd':
        for i in range(img.shape[0]):
            img[i] = (img[i] - np.mean(img[i])) / (np.std(img[i]) + 1.e-9)

    if RGB_is==True:
        img=converet_RGB(img, cmap=cmap)

    if RGB_is == False:
        img = np.expand_dims(img, axis=0)

    return img


def cwt(x, fs, totalscal, wavelet='morl'):
    if wavelet not in pywt.wavelist():
        print('小波函数名错误')
    else:
        wfc = pywt.central_frequency(wavelet=wavelet)
        a = 2 * wfc * totalscal/(np.arange(totalscal,0,-1))
        period = 1.0 / fs
        [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
        amp = abs(cwtmar)
        return amp, fre


def wave(data,resampe_is=False,samplenum=2,mask_is=False,resize_is=False,fs=128,scales=128,norm='maxmin',resize_num=128):

    sgn=data
    if resampe_is==True:
        sgn=resampe(sgn,samplenum=samplenum)
    img, fre = cwt(sgn, fs, scales, 'morl')
    img=np.mean(img,axis=1)
    # spec = PseudoWignerVilleDistribution(sgn)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    if norm=='maxmin':
        normImg = (img - img.min()) / (img.max() - img.min() +0.000001)
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'lognew':
        normImg = (np.log(np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)

    if resize_is==True:
        normImg = cv2.resize(normImg, (resize_num,resize_num))
    normImg = np.expand_dims(normImg, axis=0)
    return normImg


def wave1(data,resize_is=False,fs=1024,scales=512,norm='log'):
    sgn=data
    if sgn.shape[0]==2:
        sgn = data[0, :] + data[1, :] * 1j
        sgn=np.expand_dims(sgn,0)
    img, fre = cwt(sgn[0], fs, scales, 'morl')
    # spec = PseudoWignerVilleDistribution(sgn)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)a
    if norm=='maxmin':
        normImg = (img - img.min()) / (img.max() - img.min() +0.000001)
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'lognew':
        normImg = (np.log(np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    if resize_is==True:
        normImg = cv2.resize(normImg, (128,128))
    normImg = np.expand_dims(normImg, axis=0)
    return normImg

