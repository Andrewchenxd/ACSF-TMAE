import numpy as np
import torch
import random
import scipy.io as scio
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from scipy.signal import resample, hilbert2
from utils.signal_aug import *
from utils.signeltoimage import *
import pickle
def toIQ(sgn):
    newsgn = np.zeros((2, sgn.shape[0]))
    y = hilbert2(sgn)
    newsgn[0] = np.real(y)
    newsgn[1] = np.imag(y)
    return newsgn


def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)


# dataset:RML2016.10a (train test 7:3)
class SigDataSet_stft(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False):
        super().__init__()
        self.data = scio.loadmat(data_path)['data']
        self.labels = scio.loadmat(data_path)['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        if (adsbis == False) and (newdata == False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.rml == True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.return_label == False:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                        self.SNR1
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)
        elif self.adsbis==True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.resample_is == True:
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
                self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
            if self.return_label == False:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32)
            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)

def Img_aug(img,imgn):
    if np.random.rand() < 0.2:
        img = np.flip(img, 1).copy()
        imgn = np.flip(imgn, 1).copy()
    elif np.random.rand() < 0.4:
        img = np.flip(img, 2).copy()
        imgn = np.flip(imgn, 2).copy()
    elif np.random.rand() < 0.6:
        img = np.flip(np.flip(img, 2),1).copy()
        imgn = np.flip(np.flip(imgn, 2),1).copy()
    elif np.random.rand() < 1:
        img=img
        imgn=imgn
    return img,imgn

class SigDataSet_pwvd(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no',imgaug=False
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False, RGB_is=False,zhenshiSNR=False,freq_fliter=False):
        super().__init__()
        if adsbis == False:
            self.data = scio.loadmat(data_path)['data']
            self.labels = scio.loadmat(data_path)['label'].flatten()
        else:
            self.data = pickle.load(open(data_path, 'rb'), encoding='latin')['data']
            self.labels = pickle.load(open(data_path, 'rb'), encoding='latin')['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.sgnaug = sgnaug
        self.imgaug=imgaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is=RGB_is
        self.zhenshiSNR=zhenshiSNR
        self.freq_fliter=freq_fliter
        if (adsbis == False) and (newdata == False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            if np.random.random() <= 0.5:
                self.sgn = sig_time_warping(self.data[item])
            else:
                self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)

        np.random.seed(None)
        self.SNR = random.randint(self.snrmin, self.snrmax)
        self.SNR1 = self.SNR - 20
        if self.sgnaug == False:
            self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
        else:
            if np.random.random() <= 0.75:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
            else :
                self.sgn_noise = rayleigh_noise(self.sgn_noise, self.SNR1,self.zhenshiSNR,Seed=None)
        if self.freq_fliter:
            self.sgn_noise1 =filter(self.sgn_noise, filiter='low', filiter_threshold=0.85,
                filiter_size=0.001,middle_zero=True,freq_smooth=True,return_IQ=True)
            self.sgn_noise=self.sgn_noise1+self.sgn_noise
        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        img=pwvd(data=self.sgn, norm=self.norm, resize_is=self.resize_is,RGB_is=self.RGB_is)
        imgn=pwvd(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is,RGB_is=self.RGB_is)
        if self.imgaug == True:
            img,imgn=Img_aug(img,imgn)

        if self.return_label == False:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(img, dtype=torch.float32), \
                   torch.tensor(imgn, dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_gasf(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no',data_name='RML2016.10a'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False,RGB_is=False):
        super().__init__()
        if adsbis == False:
            self.data = scio.loadmat(data_path)['data']
            self.labels = scio.loadmat(data_path)['label'].flatten()
        else:
            self.data = pickle.load(open(data_path, 'rb'), encoding='latin')['data']
            self.labels = pickle.load(open(data_path, 'rb'), encoding='latin')['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.data_name=data_name
        self.rml = True
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        self.RGB_is=RGB_is
        if (adsbis == False) and (newdata == False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.resample_is == True:
            self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            self.sgn_noise = resampe(self.sgn_noise, samplenum=self.samplenum)
        if self.data_name == 'RMLc':
            self.sgn = self.sgn / 500
            self.sgn_noise = self.sgn_noise / 500
        np.random.seed(None)
        self.SNR = random.randint(self.snrmin, self.snrmax)
        self.SNR1 = self.SNR - 20
        # if self.sgnaug == False:
        #     self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
        # else:
        #     if np.random.random() <= 0.5:
        #         self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
        #     else:
        #         self.sgn_noise = addmask(self.sgn_noise)
        if self.return_label == False:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32)
        elif self.return_label == True:
            return torch.tensor(gasf(self.sgn, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(gasf(self.sgn_noise, self.norm, self.resize_is,RGB_is=self.RGB_is), dtype=torch.float32), \
                   torch.tensor(self.labels[item], dtype=torch.long)


class SigDataSet_wave(Dataset):
    def __init__(self, data_path, adsbis=False, newdata=False, snr_range=[1, 7], resample_is=False, samplenum=15,
                 norm='no'
                 , chazhi=False, chazhinum=2, is_DAE=False, resize_is=False, return_label=False, sgnaug=False
                 , sgn_expend=False):
        super().__init__()
        self.data = scio.loadmat(data_path)['data']
        self.labels = scio.loadmat(data_path)['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is = resample_is
        self.norm = norm
        self.resize_is = resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum = samplenum
        self.is_DAE = is_DAE
        self.rml = True
        self.sgnaug = sgnaug
        self.sgn_expend = sgn_expend
        self.return_label = return_label
        if (adsbis == False) and (newdata == False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata == True):
            self.rml = False
        self.newdata = newdata

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sgn_expend == False:
            self.sgn = self.data[item]
            self.sgn_noise = np.copy(self.sgn)
        else:
            self.sgn = sig_time_warping(self.data[item])
            self.sgn_noise = np.copy(self.sgn)
        if self.rml == True:
            np.random.seed(None)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR - 20
            if self.sgnaug == False:
                self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
            else:
                if np.random.random() <= 0.5:
                    self.sgn_noise = awgn(self.sgn_noise, self.SNR1)
                else:
                    self.sgn_noise = addmask(self.sgn_noise)
            if self.return_label == False:
                return torch.tensor(wave(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(wave(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       self.SNR1


            elif self.return_label == True:
                return torch.tensor(stp(data=self.sgn, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32), \
                       torch.tensor(stp(data=self.sgn_noise, norm=self.norm, resize_is=self.resize_is), dtype=torch.float32),\
                    torch.tensor(self.labels[item], dtype=torch.long)



