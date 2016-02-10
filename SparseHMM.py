import numpy as np
import sys

class Observer():
    def __init__(self, init, frm, to, transProb):
        self.init = init
        self.frm = frm
        self.to = to
        self.transProb = transProb
    
    def decodeViterbi(self, obsProb):
        nState = len(self.init)
        nFrame = len(obsProb)
        nTrans = len(self.transProb)
        
        scale = np.zeros((nFrame), dtype = np.float64)
        
        if(len(obsProb) < 1): # too short
            return np.array([], dtype = np.int)
        
        delta = np.zeros((nState), dtype = np.float64)
        oldDelta = np.zeros((nState), dtype = np.float64)
        psi = np.zeros((nFrame, nState), dtype = np.int) # matrix of remembered indices of the best transitions
        path = np.ndarray((nFrame), dtype = np.int) # the final output path
        path.fill(nState - 1)
        
        # init first frame
        oldDelta = self.init * obsProb[0][:len(self.init)]
        deltaSum = np.sum(oldDelta)
        
        scale[0] = 1.0 / deltaSum
        
        # rest of forward step
        
        for iFrame in range(1, nFrame):
            for iTrans in range(0, nTrans):
                fromState = self.frm[iTrans]
                toState = self.to[iTrans]
                currTransProb = self.transProb[iTrans]
                
                currValue = oldDelta[fromState] * currTransProb
                if(currValue > delta[toState]):
                    delta[toState] = currValue # will be multiplied by the right obs later
                    psi[iFrame][toState] = fromState
            
            delta *= obsProb[iFrame][:len(self.init)]
            deltaSum = np.sum(delta)
            
            if(deltaSum > 0):
                oldDelta = delta / deltaSum
                delta.fill(0)
                scale[iFrame] = 1.0 / deltaSum
            else:
                print("WARNING: Viterbi has been fed some zero probabilities at frame %d." % (iFrame), file = sys.stderr)
                oldDelta.fill(1.0 / nState)
                delta.fill(0)
                scale[iFrame] = 1.0
        
        # init backward step
        bestStateIdx = np.argmax(oldDelta)
        bestValue = oldDelta[bestStateIdx]
        path[-1] = bestStateIdx
        # rest of backward step
        for iFrame in reversed(range(nFrame - 1)):
            path[iFrame] = psi[iFrame + 1][path[iFrame + 1]]
        return path