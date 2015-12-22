import numpy as np
import YINHelper
import dists
import beta
import MonoPitch
from utils import *

class Processor():
    def __init__(self, samprate):
        self.samprate = int(samprate)
        self.threshold = 0.5
        
        self.pdf = beta.normalized_pdf(1.7, beta.au_to_b(1.7, 0.2), 0, 1, 100)
        #self.pdf = dists.betaDist2
        
        self.lowAmp = 0.1
        self.bias = 1.0
        self.hopSize = int(2 ** np.ceil(np.log2(self.samprate * 0.005)))
        self.windowSize = int(2 ** np.ceil(np.log2(self.samprate * 0.025)))
    
    def process(self, input):
        """
        input wave data, return a list of frequency in Herz
        if frame is unvoiced, freq will be less than zero
        """
        
        hops = []
        
        for iHop in range(int(len(input) / self.hopSize)):
            hop = getFrame(input, iHop * self.hopSize, self.windowSize)
            hops.append(self.processHop(hop))
        mp = MonoPitch.Processor()
        
        hops = np.array(hops)
        o = mp.process(hops)
        
        return o
    
    def processHop(self, input):
        """
        input a frame, return freqProbs
        freqProbs.shape == (nValley, 2)
        freqProbs[n][0] is frequency in Herz
        freqProbs[n][1] is probability
        """
    
        buffSize = int(len(input) / 2)
        
        buff = YINHelper.fastDifference(input, buffSize)
        buff = YINHelper.cumulativeDifference(buff)
        
        valleys = YINHelper.findValleys(buff, threshold = self.threshold)
        
        mean = np.sum(input[:buffSize]) / buffSize
        input -= mean
        
        freqProb = []
        for vidx, curr in enumerate(valleys):
            freq = self.samprate / YINHelper.parabolicInterpolation(buff, curr)
            prob = 0.0
            v0 = 1 if(vidx == 0) else (buff[valleys[vidx - 1]] + 1e-10)
            v1 = buff[curr] + 1e-10
            prob = np.sum(self.pdf[int(v1 * 100):int(v0 * 100)])
            prob = min(prob, 0.99)
            prob *= self.bias
            freqProb.append((freq, prob))
        freqProb = np.array(freqProb, dtype = np.float64)
        
        return freqProb