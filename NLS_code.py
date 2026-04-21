import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def load_data():
    data = pd.read_csv('Leigh1968_harelynx.csv')
    data['hare'] = data['hare'] / 1000
    data['lynx'] = data['lynx'] / 1000
    return data

def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def solve_model(params, data):
    alpha, beta, delta, gamma = params
    t_span = (0, len(data) - 1)
    t_eval = np.arange(len(data))
    z0 = [data['hare'].iloc[0], data['lynx'].iloc[0]]
    sol = solve_ivp(lotka_volterra, t_span, z0, args=(alpha, beta, delta, gamma), t_eval=t_eval)
    return sol.y

def loss_func(params, data):
    if any(p <= 0 for p in params):
        return 1e10
    try:
        x_model, y_model = solve_model(params, data)
        if len(x_model) != len(data):
            return 1e10
        error = np.sum((x_model - data['hare'].values)**2) + \
                np.sum((y_model - data['lynx'].values)**2)
        return error
    except Exception:
        return 1e10


if __name__ == "__main__":
    data = load_data()
    initial_params = [1.0, 0.1, 0.1, 1.0]
    bounds = [(1e-5, 10), (1e-5, 10), (1e-5, 10), (1e-5, 10)]
    result = minimize(loss_func, initial_params, args=(data,), method='L-BFGS-B', bounds=bounds)
    best_params = result.x
    print("Best parameters:", best_params)
    
    x_model, y_model = solve_model(best_params, data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['year'], data['hare'], label='Hare (Data)', marker='o')
    plt.plot(data['year'], data['lynx'], label='Lynx (Data)', marker='o')
    plt.plot(data['year'], x_model, label='Hare (NLS)', linestyle='--')
    plt.plot(data['year'], y_model, label='Lynx (NLS)', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Population (thousands)')
    plt.title('Lotka-Volterra Model Fit to Data')
    plt.legend()
    plt.grid()
    plt.savefig('lotka_volterra_fit.png')
    plt.show()

    plt.figure(figsize=(6,6))
    plt.plot(data['hare'], data['lynx'], label='Data', marker='o')
    plt.plot(x_model, y_model, label='NLS', linestyle='--')
    plt.xlabel('Hare Population')
    plt.ylabel('Lynx Population')
    plt.title('Phase Portrait: Lotka-Volterra Model')
    plt.legend()
    plt.grid()
    plt.savefig('lotka_volterra_phase_portrait.png')
    plt.show()