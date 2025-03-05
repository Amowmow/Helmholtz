import numpy as np
import matplotlib.pyplot as plt

def helm_aprox (I, radius, n):
    mew_0 = 4*np.pi*10**-7
    B = (8*mew_0*I*n)/(radius*np.sqrt(125))
    return B

B = helm_aprox(500, .3, 330)
print(B)