import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Leigh1968_harelynx.csv")


hare_data = df["hare"].values.astype(float) / 1000.0
lynx_data = df["lynx"].values.astype(float) / 1000.0
years = df["year"].values


a = 1.595489
b = 0.066076
c = 2.047119
d = 0.041158

x0 = hare_data[0]
y0 = lynx_data[0]


def euler_solver(a, b, c, d, x0, y0, h, steps):
    x = np.zeros(steps)
    y = np.zeros(steps)

    x[0] = x0
    y[0] = y0

    for i in range(1, steps):
        dx = a * x[i - 1] - b * x[i - 1] * y[i - 1]
        dy = -c * y[i - 1] + d * x[i - 1] * y[i - 1]

        x[i] = x[i - 1] + h * dx
        y[i] = y[i - 1] + h * dy

       
        if x[i] < 0:
            x[i] = 0
        if y[i] < 0:
            y[i] = 0

    return x, y


t_end = years[-1] - years[0]
h = 0.001
steps = int(t_end / h) + 1


x_model, y_model = euler_solver(a, b, c, d, x0, y0, h, steps)


plt.figure(figsize=(7, 7))


plt.plot(
    hare_data,
    lynx_data,
    'o-',
    linewidth=1,
    markersize=4,
    label="Observed Data"
)

plt.plot(x_model[:8000], y_model[:8000], '--', linewidth=2, label="Hamiltonian Model")

plt.xlabel("Hare Population (thousands)")
plt.ylabel("Lynx Population (thousands)")
plt.title("Phase Portrait Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("phase_portrait_comparison.png", dpi=300)
plt.show()