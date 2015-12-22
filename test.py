import YINHelper
import PYIN
import dists
import numpy as np
import pylab as pl
import beta
import wave
import MonoPitch
from utils import *
from scipy import signal

f = wave.open("orig.wav", "rb")
params = f.getparams()
ch, sw, sr, frameCount = params[:4]
if(ch > 1):
    print("DO NOT USE MULTICHANNEL SOURCE!")
    sys.exit(1)
str_data = f.readframes(frameCount)
f.close()

w = np.fromstring(str_data, dtype = np.short)
del str_data
w = w.astype(np.float64) / 32768.0

yin = PYIN.Processor(sr)

yin.bias = 2.0
p = yin.process(w)

# to fft bins for ploting
p = p / (sr / 2) * 1024

fft = np.zeros((int(len(w) / yin.hopSize), 1024))
for iHop in range(int(len(w) / yin.hopSize)):
    fft[iHop] = np.abs(np.fft.fft(getFrame(w, iHop * yin.hopSize, 2048) * signal.blackmanharris(2048)))[0:1024]
fft = np.log10(fft.T)[:256] # [0, 256) bin for better ploting

pl.xlabel("Frame")
pl.ylabel("Freq")
pl.imshow(fft, origin = 'lower', cmap = 'jet', interpolation = 'bicubic', aspect = 'auto')
pl.plot(np.arange(len(p)), p, 'ro', np.arange(len(p)), p)

pl.show()
