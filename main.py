import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-3

def Q(u):
    a1 = 0.5
    a2 = 0.25
    b1 = 1.00
    b2 = 1.00

    k11 = (-b1) / (a1 * a2 - 1)
    k12 = (-a1 * b2) / (a1 * a2 - 1)
    k21 = (-a2 * b1) / (a1 * a2 - 1)
    k22 = (-b2) / (a1 * a2 - 1)

    y1 = k11 * u[0] + k12 * u[1] + np.random.triangular(-0.001,0,0.001)
    y2 = k21 * u[0] + k22 * u[1] + np.random.triangular(-0.001,0,0.001)

    Q = (y1 - 4) ** 2 + (y2 - 4) ** 2

    return Q


# Algorytm optymalizacji dwupoziomowej
def halving_method(Q, a, b, epsilon):
    while (b - a) > epsilon:
        mid = (a + b) / 2

        if Q(mid - epsilon) < Q(mid + epsilon):
            b = mid
        else:
            a = mid

    return (a + b) / 2


# Algorytm optymalizacji dwupoziomowej
def optimal_points():
    # Optimize for u1 and u2 at the border [-1, 1]
    u1_optimal = halving_method(lambda u1: Q([u1, np.sqrt(1 - u1 ** 2)]), -1, 1, epsilon)
    u2_optimal = halving_method(lambda u2: Q([np.sqrt(1 - u2 ** 2), u2]), -1, 1, epsilon)

    optimal_point = np.array([u1_optimal, u2_optimal])
    minimal_value = Q(optimal_point)

    return minimal_value, optimal_point


def generate_graph():
    circle = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(circle)
    circle_y = np.sin(circle)

    plt.figure(figsize=(8, 8))
    plt.plot(circle_x, circle_y, label='Okrąg jednostkowy', linestyle='dotted', color='gray')
    plt.scatter(2, 3, color='red', label='Idealne u', s=50)
    plt.scatter(optimal_point[0], optimal_point[1], color='green', label='Optymalne u', s=50)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(-1, 2.5, 0.5), labels=['-1.0', '-0.5', '0.0', '0.5', '1.0', '1.5', '2.0'])
    plt.yticks(np.arange(-0.5, 3.6, 0.5), labels=['-0.5', '0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'])
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    plt.text(2, 3, 'Idealne u', verticalalignment='bottom', horizontalalignment='right', color='red')
    plt.text(optimal_point[0], optimal_point[1], 'Optymalne u', verticalalignment='bottom', horizontalalignment='right',
             color='green')
    plt.show()


minimal_value, optimal_point = optimal_points()
generate_graph()

a1 = 0.5
a2 = 0.25
b1 = 1.0
b2 = 1.0

k11 = (-b1) / (a1 * a2 - 1)
k12 = (-a1 * b2) / (a1 * a2 - 1)
k21 = (-a2 * b1) / (a1 * a2 - 1)
k22 = (-b2) / (a1 * a2 - 1)

y1 = k11 * optimal_point[0] + k12 * optimal_point[1]
y2 = k21 * optimal_point[0] + k22 * optimal_point[1]

min_avg=0
opt_avg=0

for i in range(0,100):
    minimal_value, optimal_point = optimal_points()

print("Minimalna wartość funkcji celu:", round(minimal_value, 4))
print("Optymalny punkt u1:", round(optimal_point[0], 4))
print("Optymalny punkt u2:", round(optimal_point[1], 4))
print("Optymalne wartości:")
print("y1 =", round(y1, 3))
print("y2 =", round(y2, 3))
print("k11 =", round(k11, 3))
print("k12 =", round(k12, 3))
print("k21 =", round(k21, 3))
print("k22 =", round(k22, 3))
