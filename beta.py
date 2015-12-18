import numpy as np

def au_to_b(a, u):
    return a / u - a;

def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype = np.float64) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    return v / np.sum(v)