import numpy as np

def getFrame(input, center, size):
    out = np.zeros((size), input.dtype)
    leftSize = int(size / 2)
    rightSize = size - leftSize # for odd size
    
    inputBegin = max(center - leftSize, 0)
    inputEnd = min(center + rightSize, len(input))
    
    outBegin = -min(center - leftSize, 0)
    outEnd = outBegin + (inputEnd - inputBegin)
    
    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out