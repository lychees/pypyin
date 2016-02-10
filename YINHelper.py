import numpy as np
import sys

def slowDifference(input, outSize):
    outSize = int(outSize)
    out = np.zeros((outSize), dtype = input.dtype)
    for i in range(1, outSize):
        start = int(outSize / 2 - i / 2)
        end = int(start + outSize)
        for j in range(start, end):
            delta = input[i + j] - input[j]
            out[i] += delta * delta
    return out

def fastDifference(input, outSize):
    outSize = int(outSize)
    out = np.zeros((outSize), dtype = input.dtype)

    frameSize = outSize * 2
    
    # POWER TERM CALCULATION
    # ... for the power terms in equation (7) in the Yin paper
    
    powerTerms = np.zeros((outSize), dtype = np.float64)
    powerTerms[0] = np.sum(input[:outSize] ** 2)
    
    # now iteratively calculate all others (saves a few multiplications)
    for i in range(1, outSize):
        powerTerms[i] = powerTerms[i - 1] - (input[i - 1] ** 2) + input[i + outSize] * input [i + outSize]
    
    # YIN-STYLE ACF via FFT
    # 1. data
    transformedAudio = np.fft.rfft(input)
    
    # 2.half of the data, disguised as a convolution kernel
    kernel = np.zeros((frameSize), dtype = np.float64)
    kernel[:outSize] = input[:outSize][::-1]
    transformedKernel = np.fft.rfft(kernel)
    
    # 3. convolution
    yinStyleACF = transformedAudio * transformedKernel
    transformedAudio = np.fft.irfft(yinStyleACF)
    
    # CALCULATION OF difference function
    # according to (7) in the Yin paper
    out = powerTerms[0] + powerTerms[:outSize] - 2 * transformedAudio.real[outSize - 1:-1]
    return out

def cumulativeDifference(input):
    out = input.copy()
    out[0] = 1.0
    sum = 0.0
    
    for i in range(1, len(out)):
        sum += out[i]
        if(sum == 0):
            out[i] = 1
        else:
            out[i] *= i / sum
    return out
        
def findValleys(x, threshold = 0.5, step = 0.01):
    ret = []
    for i in range(1, len(x) - 1):
        prev = x[i - 1]
        curr = x[i]
        next = x[i + 1]
        if(prev > curr and next > curr and curr < threshold):
            threshold = curr - step
            ret.append(i)
    return ret

def parabolicInterpolation(input, i):
    lin = len(input)
    if(i == lin): # bad value
        return float(idx)
    
    ret = 0.0
    if(i > 0 and i < lin - 1):
        s0 = float(input[i - 1])
        s1 = float(input[i])
        s2 = float(input[i + 1])
        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
        if(abs(adjustment) > 1):
            adjustment = 0.0
        ret = i + adjustment
    else:
        ret = i
    
    return ret