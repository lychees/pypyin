import YINHelper
import PYIN
import dists
import numpy as np
import pylab as pl
import beta
import wave
import MonoPitch

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

frameSize = 512

fft = np.zeros((int(len(w) / frameSize), 2000))
for iFrame in range(int(len(w) / frameSize)):
    ffted = np.abs(np.fft.fft(w[iFrame * frameSize:(iFrame + 1) * frameSize]))[0:frameSize / 2]
    fpb = (sr / 2.0) / (frameSize / 2.0)
    ffted_f = np.zeros((sr / 2))
    for bin, val in enumerate(ffted):
        ffted_f[bin * fpb:(bin + 1) * fpb] = val
    fft[iFrame] = 20 * np.log10(ffted_f[0:2000])
fft = fft.T

yin = PYIN.Processor(sr)

yin.bias = 1.0
o = yin.process(w, frameSize)

yin.bias = 2.0
p = yin.process(w, frameSize)

pl.xlabel("Frame")
pl.ylabel("Freq")
pl.imshow(fft, origin = 'lower', cmap = 'jet', interpolation = 'bicubic', aspect = 'auto', vmin = 0, vmax = 100.0)
pl.plot(np.arange(len(o)), o, 'bo', np.arange(len(o)), o)
pl.plot(np.arange(len(p)), p, 'ro', np.arange(len(p)), p)

pl.show()
