import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Leigh1968_harelynx.csv")


years = df["year"].values
hare_data = df["hare"].values.astype(float)
lynx_data = df["lynx"].values.astype(float)


hare_data = hare_data / 1000.0
lynx_data = lynx_data / 1000.0


t_data = years - years[0]


a = 1.595489
b = 0.066076
c = 2.047119
d = 0.062658



x0 = hare_data[0]
y0 = lynx_data[0]


def euler_solver(a, b, c, d, x0, y0, t0, t_end, h):
    n_steps = int((t_end - t0) / h) + 1

    t = np.zeros(n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    t[0] = t0
    x[0] = x0
    y[0] = y0

    for i in range(1, n_steps):
        dx = a * x[i-1] - b * x[i-1] * y[i-1]
        dy = -c * y[i-1] + d * x[i-1] * y[i-1]

        x[i] = x[i-1] + h * dx
        y[i] = y[i-1] + h * dy
        t[i] = t[i-1] + h

    return t, x, y


t0 = 0
t_end = t_data[-1]
h = 0.01

t_model, x_model, y_model = euler_solver(a, b, c, d, x0, y0, t0, t_end, h)


x_at_data = np.interp(t_data, t_model, x_model)
y_at_data = np.interp(t_data, t_model, y_model)


plt.figure(figsize=(10, 6))
plt.plot(years, hare_data, 'o', label="Observed Hare")
plt.plot(years, lynx_data, 'o', label="Observed Lynx")
plt.plot(years, x_at_data, '-', label="Model Hare")
plt.plot(years, y_at_data, '-', label="Model Lynx")

plt.xlabel("Year")
plt.ylabel("Population (thousands)")
plt.title("Observed Data vs Model Prediction")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(hare_data, lynx_data, 'o-', label="Observed Data")
plt.xlabel("Hare Population (thousands)")
plt.ylabel("Lynx Population (thousands)")
plt.title("Observed Phase Portrait")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(x_model, y_model, '-', label="Model Trajectory")
plt.xlabel("Hare Population (thousands)")
plt.ylabel("Lynx Population (thousands)")
plt.title("Model Phase Portrait")
plt.legend()
plt.tight_layout()
plt.savefig("model_phase_portrait.png", dpi=300)
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(hare_data, lynx_data, 'o-', label="Observed Data")
plt.plot(x_model, y_model, '-', label="Model Trajectory")
plt.xlabel("Hare Population (thousands)")
plt.ylabel("Lynx Population (thousands)")
plt.title("Observed vs Model Phase Portrait")
plt.legend()
plt.tight_layout()
plt.savefig("observed_vs_model_phase_portrait.png", dpi=300)
plt.show()