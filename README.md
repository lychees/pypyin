# pypyin
An PYIN F0 estimator implement in Python

## Requirement
Python 3, Numpy, Pylab

## Usage
test.py: An example program.

PYIN.py: PYin processor
> PYin.process(input, frameSize): input wave data and frameSize, return a list of freq
>>  *if frame is unvoiced, freq will be less than zero*

> PYin.processFrame(input): input a frame, return freqProb
