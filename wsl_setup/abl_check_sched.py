import numpy as np

eta = 0.1


def exp_sched(t):
    return eta * np.exp(10 * (t - 1))


def quad_sched(t):
    return np.sqrt(max(0.0, (eta ** 2 - 2 * eta) * t ** 2 + 2 * eta * t))


for name, f in [('exp', exp_sched), ('quad', quad_sched)]:
    print(name, [round(float(f(t)), 5) for t in [0.0, 0.25, 0.5, 0.75, 1.0]])
