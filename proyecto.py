
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
        t = 3*(1-u)/u
        integral += fun(t) / (u**2)
    return integral / N


    
def ejecucion():
    N = np.linspace(1, 500000, 50)  # 50 valores entre 1 y 500.000
    Y = [aprox_Monte(int(n)) for n in N]
    plt.plot(N, [real_value]*50, label='Valor Real')
    plt.plot(N, Y)
    plt.xlabel('Número de Muestras')
    plt.ylabel('Estimación de P(X > 3)')
    plt.title('Método de Monte Carlo para P(X > 3)')
    
    plt.legend()
    plt.show()
    return

real_value = 1 - stats.norm.cdf(3)

print(real_value)
print(aprox_Monte(100000))
    

    
def importance_sampling(N, imp_fun):
    integral = 0
    for n in range(N):
        u = random.random()
        t = imp_fun(u)
        integral += fun(t) / (imp_fun(u))
    return integral / N
    
    # Función de importancia 1: Y ~ N(4, 1)
def imp_fun1(u):
    return np.random.normal(4, 1)

# Función de importancia 2: Y ~ Exp(1/4)
def imp_fun2(u):
    return np.random.exponential(4)

# Función de importancia 3: Y ~ Uniforme(3, 7)
def imp_fun3(u):
    return 3 + (7-3)*u

# Ejecución del método de Importance Sampling
def ejecucion_importance_sampling():
    N = np.linspace(1, 500000, 50)  # 50 valores entre 1 y 500.000
    
    Y1 = [importance_sampling(int(n), imp_fun1) for n in N]
    Y2 = [importance_sampling(int(n), imp_fun2) for n in N]
    Y3 = [importance_sampling(int(n), imp_fun3) for n in N]
    
    #valor real
    real_value = 1 - stats.norm.cdf(3)
    
    plt.plot(N, [real_value]*50, label='Valor Real')
    plt.plot(N, Y1, label='N(4, 1)')
    plt.plot(N, Y2, label='Exp(1/4)')
    plt.plot(N, Y3, label='Uniforme(3, 7)')
    plt.xlabel('Número de Muestras')
    plt.ylabel('Estimación de P(X > 3)')
    plt.title('Método de Importance Sampling para P(X > 3)')
    plt.legend()
    plt.show()
    return

# Ejecutar la aproximación de Importance Sampling y graficar


print(importance_sampling(100000, imp_fun1))
print(importance_sampling(100000, imp_fun2))
print(importance_sampling(100000, imp_fun3))


# Parámetros del problema
lambda_call = 2  # tasa de llamadas por minuto
target_calls = 9
time_limit = 10

# Función para generar eventos Poisson
def eventosPoisson(lamda, T):
    t = 0
    NT = 0
    Eventos = []
    while t < T:
        U = 1 - random.random()
        t += -np.log(U) / lamda
        if t <= T:
            NT += 1
            Eventos.append(t)
    return NT, Eventos

# Método tradicional de Monte Carlo usando eventos Poisson
def monte_carlo_calls_poisson(N):
    count = 0
    for _ in range(N):
        NT, Eventos = eventosPoisson(lambda_call, 1)
        if NT >= target_calls:
            count += 1
    return count / N

# Ejecución del método Monte Carlo tradicional con eventos Poisson
def ejecucion_monte_carlo_poisson():
    N= list(range(10000, 500000, 10000))
    results = [monte_carlo_calls_poisson(int(n)) for n in N]
    
    plt.plot(N, results, label='Monte Carlo (Poisson)')
    plt.xlabel('Número de Muestras')
    plt.ylabel('Estimación de Probabilidad')
    plt.title('Probabilidad de Esperar al Menos 10 Minutos para 9 Llamadas')
    plt.axhline(y=1 - stats.gamma(a=9, scale=0.5).cdf(10), color='r', linestyle='-', label='Valor Real')
    plt.legend()
    plt.show()
    return

# Ejecutar la aproximación Monte Carlo tradicional y graficar
ejecucion_monte_carlo_poisson()

def importance_sampling_calls(N, imp_fun):
    integral = 0
    for n in range(N):
        NT, Eventos = eventosPoisson(lambda_call, 1)
        if NT >= target_calls:
            u = random.random()
            t = imp_fun(u)
            integral += fun(t) / (imp_fun(u))
    return integral / N

# Ejecución del método de Importance Sampling para el problema de llamadas
def ejecucion_importance_sampling_calls():
    N= list(range(10000, 500000, 10000))
    
    Y1 = [importance_sampling_calls(int(n), imp_fun1) for n in N]
    Y2 = [importance_sampling_calls(int(n), imp_fun2) for n in N]
    Y3 = [importance_sampling_calls(int(n), imp_fun3) for n in N]
    
    plt.plot(N, Y1, label='N(4, 1)')
    plt.plot(N, Y2, label='Exp(1/4)')
    plt.plot(N, Y3, label='Uniforme(3, 7)')
    plt.xlabel('Número de Muestras')
    plt.ylabel('Estimación de Probabilidad')
    plt.title('Método de Importance Sampling para Esperar al Menos 10 Minutos para 9 Llamadas')
    plt.axhline(y=1 - stats.gamma(a=9, scale=0.5).cdf(10), color='r', linestyle='-', label='Valor Real')
    plt.legend()
    plt.show()
    return

# Ejecutar la aproximación de Importance Sampling para el problema de llamadas y graficar


ejecucion_importance_sampling_calls()