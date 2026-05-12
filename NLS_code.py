import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

HAM_PARAMS = np.array([1.595489, 0.066076, 0.041158, 2.047119])

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

def log_posterior(params, data):
    alpha, beta, delta, gamma = params
    if alpha <= 0 or beta <= 0 or delta <= 0 or gamma <= 0:
        return -np.inf
    try:
        x_model, y_model = solve_model(params, data)
        if len(x_model) != len(data):
            return -np.inf
        hare_obs = data['hare'].values
        lynx_obs = data['lynx'].values
        sigma_hare = 5.0
        sigma_lynx = 5.0
        log_like = -0.5 * (
            np.sum(((hare_obs - x_model) / sigma_hare)**2) +
            np.sum(((lynx_obs - y_model) / sigma_lynx)**2)
        )
        return log_like
    except Exception:
        return -np.inf

def run_mcmc(initial_params, data, n_samples=5000, proposal_scale=0.02):
    current = np.array(initial_params)
    current_log_post = log_posterior(current, data)
    samples = []
    accepted = 0
    for _ in range(n_samples):
        proposal = current + np.random.normal(0, proposal_scale, size=4)
        proposal_log_post = log_posterior(proposal, data)
        log_accept_ratio = proposal_log_post - current_log_post
        if np.log(np.random.rand()) < log_accept_ratio:
            current = proposal
            current_log_post = proposal_log_post
            accepted += 1
        samples.append(current.copy())
    acceptance_rate = accepted / n_samples
    print("Acceptance rate:", acceptance_rate)
    return np.array(samples)

def equilibrium_point(params):
    alpha, beta, delta, gamma = params
    x_star = gamma / delta
    y_star = alpha / beta
    return x_star, y_star

def jacobian_matrix(params):
    alpha, beta, delta, gamma = params
    x_star, y_star = equilibrium_point(params)
    J = np.array([
        [alpha - beta * y_star, -beta * x_star],
        [delta * y_star,        delta * x_star - gamma]
    ])
    return J

def compute_eigenvals(params):
    J = jacobian_matrix(params)
    eigvals = np.linalg.eigvals(J)
    return eigvals

def oscillation_freq(params):
    alpha, _, _, gamma = params
    return np.sqrt(alpha * gamma)

if __name__ == "__main__":
    data = load_data()
    initial_params = [1.0, 0.1, 0.1, 1.0]
    bounds = [(1e-5, 10), (1e-5, 10), (1e-5, 10), (1e-5, 10)]
    result = minimize(loss_func, initial_params, args=(data,), method='L-BFGS-B', bounds=bounds)
    best_params = result.x
    print("Best NLS parameters:", best_params)

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

    plt.figure(figsize=(6, 6))
    plt.plot(data['hare'], data['lynx'], label='Data', marker='o')
    plt.plot(x_model, y_model, label='NLS', linestyle='--')
    plt.xlabel('Hare Population')
    plt.ylabel('Lynx Population')
    plt.title('Phase Portrait: Lotka-Volterra Model')
    plt.legend()
    plt.grid()
    plt.savefig('lotka_volterra_phase_portrait.png')
    plt.show()

    print("\nRunning MCMC from NLS parameters...")
    nls_samples = run_mcmc(best_params, data, n_samples=5000, proposal_scale=0.02)
    burn_in = 1000
    nls_posterior = nls_samples[burn_in:]

    print("\nRunning MCMC from Hamiltonian parameters...")
    ham_samples = run_mcmc(HAM_PARAMS, data, n_samples=5000, proposal_scale=0.02)
    ham_posterior = ham_samples[burn_in:]

    param_names = ['alpha', 'beta', 'delta', 'gamma']
    print("\n===== Parameter Comparison =====")
    header = f"{'Param':<8} {'NLS pt':>10} {'Ham pt':>10} "
    header += f"{'NLS-MCMC mean':>14} {'NLS 95% CI':>20} "
    header += f"{'Ham-MCMC mean':>14} {'Ham 95% CI':>20}"
    print(header)
    for i, name in enumerate(param_names):
        nls_mean = np.mean(nls_posterior[:, i])
        nls_lo   = np.percentile(nls_posterior[:, i], 2.5)
        nls_hi   = np.percentile(nls_posterior[:, i], 97.5)
        ham_mean = np.mean(ham_posterior[:, i])
        ham_lo   = np.percentile(ham_posterior[:, i], 2.5)
        ham_hi   = np.percentile(ham_posterior[:, i], 97.5)
        print(
            f"{name:<8} {best_params[i]:>10.4f} {HAM_PARAMS[i]:>10.4f} "
            f"{nls_mean:>14.4f} [{nls_lo:.4f}, {nls_hi:.4f}]   "
            f"{ham_mean:>14.4f} [{ham_lo:.4f}, {ham_hi:.4f}]"
        )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, name in enumerate(param_names):
        for col, (post, label, color) in enumerate([
            (nls_posterior, 'NLS-MCMC', 'steelblue'),
            (ham_posterior, 'Ham-MCMC', 'darkorange'),
        ]):
            ax = axes[col][i]
            ax.hist(post[:, i], bins=40, color=color, alpha=0.8)
            ax.set_title(f'{label}: {name}')
            ax.grid(True)
    plt.tight_layout()
    plt.savefig('parameter_posteriors.png')
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, posterior, label in [
        (axes[0], nls_posterior, 'NLS-MCMC'),
        (axes[1], ham_posterior, 'Ham-MCMC'),
    ]:
        real_parts, imag_parts = [], []
        for params in posterior:
            for eig in compute_eigenvals(params):
                real_parts.append(np.real(eig))
                imag_parts.append(np.imag(eig))
        ax.scatter(real_parts, imag_parts, s=4, alpha=0.3)
        ax.axvline(0, linestyle='--', color='red')
        ax.set_xlabel('Re(λ)')
        ax.set_ylabel('Im(λ)')
        ax.set_title(f'Jacobian Eigenvalues — {label}')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('jacobian_eigenvalues.png')
    plt.show()

    nls_freqs = [oscillation_freq(p) for p in nls_posterior]
    ham_freqs = [oscillation_freq(p) for p in ham_posterior]

    nls_pt_freq = oscillation_freq(best_params)
    ham_pt_freq = oscillation_freq(HAM_PARAMS)

    plt.figure(figsize=(9, 5))
    plt.hist(nls_freqs, bins=40, alpha=0.6, color='steelblue', label='NLS-MCMC')
    plt.hist(ham_freqs, bins=40, alpha=0.6, color='darkorange', label='Ham-MCMC')
    plt.axvline(nls_pt_freq, color='steelblue', linestyle='--',
                label=f'NLS point ({nls_pt_freq:.3f})')
    plt.axvline(ham_pt_freq, color='darkorange', linestyle='--',
                label=f'Ham point ({ham_pt_freq:.3f})')
    plt.xlabel('|Im(λ)| = √(αγ)   [oscillation frequency at equilibrium]')
    plt.ylabel('Count')
    plt.title('Distance from Degenerate Centre (Bifurcation Boundary)\nNLS-MCMC vs Hamiltonian-MCMC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('eigenvalue_comparison.png')
    plt.show()

    print("\n===== Oscillation Frequency |Im(λ)| = sqrt(αγ) =====")
    print(f"NLS-MCMC : mean={np.mean(nls_freqs):.4f},  std={np.std(nls_freqs):.4f},  "
          f"95% CI=[{np.percentile(nls_freqs,2.5):.4f}, {np.percentile(nls_freqs,97.5):.4f}]")
    print(f"Ham-MCMC : mean={np.mean(ham_freqs):.4f},  std={np.std(ham_freqs):.4f},  "
          f"95% CI=[{np.percentile(ham_freqs,2.5):.4f}, {np.percentile(ham_freqs,97.5):.4f}]")
    higher = "NLS" if np.mean(nls_freqs) > np.mean(ham_freqs) else "Hamiltonian"
    print(f"\n=> {higher} parameters produce a higher oscillation frequency on average,")
    print("   meaning they sit farther from the Hopf bifurcation boundary.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, posterior, label, color in [
        (axes[0], nls_posterior, 'NLS-MCMC', 'steelblue'),
        (axes[1], ham_posterior, 'Ham-MCMC', 'darkorange'),
    ]:
        x_stars = [equilibrium_point(p)[0] for p in posterior]
        y_stars = [equilibrium_point(p)[1] for p in posterior]
        ax.scatter(x_stars, y_stars, s=5, alpha=0.4, color=color)
        ax.set_xlabel('Equilibrium Hare Population')
        ax.set_ylabel('Equilibrium Lynx Population')
        ax.set_title(f'Posterior Equilibrium Points — {label}')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('equilibrium_distribution.png')
    plt.show()
