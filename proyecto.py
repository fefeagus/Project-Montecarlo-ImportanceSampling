import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random as random

def fun(x):
    return  (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
    

def aprox_Monte(N):
    integral = 0
    for n in range(N):
        u = random.random()
        # transformar al intervalo (3,inf)
        t = (3/u)
        integral += fun(t) / (u**2)
    return integral / N

def ejecucion():
    # 50 valores entre 1 y 500.000
    N = np.linspace(1, 500000, 50)
    # ejecurarlo para cada valor de N
    Y = [aprox_Monte(int(n)) for n in N]
    # graficar
    plt.plot(N, Y)
    plt.show()
    return

real_value = 1 - stats.norm.cdf(3)

print(real_value)
ejecucion()
    
# funcion de importancia
def imp_fun(x):
    return (1 / np.log(x + 1)) / (x**2)
    
def importance_sampling(N):
    integral = 0
    for n in range(N):
        u = random.random()
        t = (3/u)
        integral += fun(t) / (u**2)
    return integral / N
    
    