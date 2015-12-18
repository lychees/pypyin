import numpy as np
import YINHelper
import dists
import beta
import dists
import MonoPitch

class PYin():
    def __init__(self, samprate, threshold = 0.2):
        self.samprate = int(samprate)
        self.threshold = float(threshold)
        
        self.pdf = beta.normalized_pdf(1.7, beta.au_to_b(1.7, 0.2), 0, 1, 100)
        #self.pdf = dists.betaDist2
        
        self.lowAmp = 0.1
        self.bias = 1.0
    
    def process(self, input, frameSize):
        nFrame = int(len(input) / frameSize)
        frames = []
        for iFrame in range(nFrame):
            frames.append(self.processFrame(input[iFrame * frameSize:(iFrame + 1) * frameSize]))
        mp = MonoPitch.Processor()
        o = mp.process(frames)
        return o
    
    def processFrame(self, input):
        buffSize = int(len(input) / 2)
        
        buff = YINHelper.fastDifference(input, buffSize)
        buff = YINHelper.cumulativeDifference(buff)
        
        peakProb = YINHelper.peakProb(buff, self.pdf)
        probSum = np.sum(peakProb)
        
        valleys = YINHelper.findValleys(buff)
        
        mean = np.sum(input[:buffSize]) / buffSize
        input -= mean
        
        freqProb = []
        for vidx, curr in enumerate(valleys):
            freq = min(880, self.samprate / YINHelper.parabolicInterpolation(buff, curr))
            prob = 0.0
            v0 = 1 if(vidx == 0) else (buff[valleys[vidx - 1]] + 1e-10)
            v1 = buff[curr] + 1e-10
            prob = np.sum(self.pdf[int(v1 * 100):int(v0 * 100)])
            prob = min(prob, 0.99)
            prob *= self.bias
            freqProb.append((freq, prob))
        freqProb = np.array(freqProb, dtype = np.float64)
        
        return freqProb