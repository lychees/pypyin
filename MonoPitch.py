import numpy as np
import SparseHMM

class Processor():
    def __init__(self, minFreq = 61.735, nBPS = 5, nPitch = 69, transRange = 2, transSelf = 0.99, yinTrust = 0.5):
        self.hmm = MonoPitchHMM(minFreq, nBPS, nPitch, transRange, transSelf, yinTrust)
        
    def process(self, input): 
        """
        input freqProbs, return a list of frequency in Herz
        """
        
        obsProb = np.zeros((len(input), self.hmm.obsProbArraySize()), dtype = np.float64)
        
        for iFrame in range(len(input)):
            obsProb[iFrame] = self.hmm.calcObsProb(input[iFrame])
        path = self.hmm.decodeViterbi(obsProb)
        
        out = np.zeros((len(path)), dtype = np.float64)
        
        for iFrame in range(len(path)):
            hmmFreq = self.hmm.freqs[path[iFrame]]
            bestFreq = 0.0
            leastDist = 1e+4
            if(hmmFreq > 0):
                for frame in input[iFrame]:
                    freq = frame[0]
                    dist = np.abs(hmmFreq - freq)
                    if(dist < leastDist):
                        leastDist = dist
                        bestFreq = freq
            else:
                bestFreq = hmmFreq
            if(abs(bestFreq - hmmFreq) > 100.0):
                bestFreq = hmmFreq
            out[iFrame] = bestFreq
        for iFrame in range(1, len(out)):
            if(out[iFrame - 1] <= 0.0 and out[iFrame] > 0.0):
                out[iFrame - 1] = out[iFrame]
        return out

class MonoPitchHMM(SparseHMM.Observer):
    def __init__(self, minFreq = 61.735, nBPS = 5, nPitch = 69, transRange = 2, transSelf = 0.99, yinTrust = 0.5): # BPS: bin per semitone
        self.minFreq = minFreq
        self.nBPS = nBPS
        self.nPitch = nPitch * nBPS
        self.transRange = int(transRange) * nBPS + 1 # to bins
        self.transSelf = transSelf
        self.yinTrust = yinTrust
        self.freqs = np.zeros((2 * self.nPitch), dtype = np.float64)
        self.freqs[:self.nPitch] = self.minFreq * np.power(2, np.arange(0, self.nPitch) * 1.0 / (12 * self.nBPS))
        self.freqs[self.nPitch:] = -self.freqs[:self.nPitch]
        
        # build
        halfTransRange = int(self.transRange / 2)
        #transProbSize = (halfTransRange * (halfTransRange + 1 + halfTransRange * 2) / 2) * 2 * 4 + (self.nPitch - halfTransRange * 2) * (halfTransRange * 2 + 1) * 4
        iSize = 4 * (self.nPitch * (2 * halfTransRange + 1) - halfTransRange * (halfTransRange + 1))
        self.init = np.ndarray((2 * self.nPitch), dtype = np.float64)
        self.frm = np.zeros((iSize), dtype = np.int)
        self.to = np.zeros((iSize), dtype = np.int)
        self.transProb = np.zeros((iSize), dtype = np.float64)
        
        # initial vector
        self.init.fill(self.nPitch * 0.5)
        # transitions
        
        iA = 0
        for iPitch in range(self.nPitch):
            theoreticalMinNextPitch = iPitch - halfTransRange
            minNextPitch = max(iPitch - halfTransRange, 0)
            maxNextPitch = min(iPitch + halfTransRange, self.nPitch - 1)
            
            weightSum = 0.0
            
            weights = np.zeros((maxNextPitch - minNextPitch + 1), dtype = np.float64)
            for i in range(minNextPitch, maxNextPitch + 1):
                if(i <= iPitch):
                    weights[i - minNextPitch] = i - theoreticalMinNextPitch + 1.0
                else:
                    weights[i - minNextPitch] = iPitch - theoreticalMinNextPitch + 1.0 - (i - iPitch)
            weightSum += np.sum(weights)
            
            # trans to close pitch
            for i in range(minNextPitch, maxNextPitch + 1):
                self.frm[iA] = iPitch
                self.to[iA] = i
                self.transProb[iA] = weights[i - minNextPitch] / weightSum * self.transSelf

                self.frm[iA + 1] = iPitch
                self.to[iA + 1] = i + self.nPitch
                self.transProb[iA + 1] = weights[i - minNextPitch] / weightSum * (1.0 - self.transSelf)

                self.frm[iA + 2] = iPitch + self.nPitch
                self.to[iA + 2] = i + self.nPitch
                self.transProb[iA + 2] = weights[i - minNextPitch] / weightSum * self.transSelf
                
                self.frm[iA + 3] = iPitch + self.nPitch
                self.to[iA + 3] = i
                self.transProb[iA + 3] = weights[i - minNextPitch] / weightSum * (1.0 - self.transSelf)
                iA += 4
    
    def obsProbArraySize(self):
        return 2 * self.nPitch + 1
    
    def calcObsProb(self, input): # input: [state(freq in Herz), prob]
        out = np.zeros((2 * self.nPitch + 1), dtype = np.float64)
        probYinPitched = 0.0
        
        # bin the pitches
        for freq, prob in input:
            if(freq <= self.minFreq):
                continue
            d = 0.0
            oldd = 1000.0
            for iPitch in range(0, self.nPitch):
                d = np.abs(freq - self.freqs[iPitch])
                if (oldd < d and iPitch > 0):
                    # previous bin must have been the closest
                    out[iPitch - 1] = prob
                    probYinPitched += out[iPitch - 1]
                    break
                oldd = d
        
        probReallyPitched = self.yinTrust * probYinPitched
        
        for iPitch in range(self.nPitch):
            if(probYinPitched > 0):
                out[iPitch] *= probReallyPitched / probYinPitched
            out[iPitch + self.nPitch] = (1.0 - probReallyPitched) / self.nPitch
        for i in range(len(out)):
            out[i] = max(out[i], 0.0) + 1e-05
        return out