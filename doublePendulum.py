import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation

sys.setrecursionlimit(10000)

length1 = 1
length2 = 1
mass1 = 1000
mass2 = 1000
gravity = 100
dt = 0.01
final_time = 10
steps = final_time / dt

theta1_0 = np.radians(60)
omega1_0 = 0
theta2_0 = np.radians(30)
omega2_0 = 0

def coordinates(theta1, theta2):
    x1 = length1 * np.sin(theta1)
    y1 = -length1 * np.cos(theta1)
    x2 = x1 + length2 * np.sin(theta2)
    y2 = y1 - length2 * np.cos(theta2)
    return x1, y1, x2, y2

angles1 = [theta1_0]
velocities1 = [omega1_0]
angles2 = [theta2_0]
velocities2 = [omega2_0]

x1_0, y1_0, x2_0, y2_0 = coordinates(theta1_0, theta2_0)
KE1 = 0.5 * mass1 * (length1 * omega1_0) ** 2
KE2 = 0.5 * mass2 * ((length1 * omega1_0) ** 2 + (length2 * omega2_0) ** 2 + 2 * length1 * length2 * omega1_0 * omega2_0 * np.cos(theta1_0 - theta2_0))
PE1 = mass1 * gravity * y1_0
PE2 = mass2 * gravity * y2_0

KE = [KE1 + KE2]
PE = [PE1 + PE2]
actualME = KE[0] + PE[0]
ME = [KE[0] + PE[0]]
error = [ME[0] - actualME]

def simulate(index, theta1_prev, omega1_prev, theta2_prev, omega2_prev):
    if index >= steps:
        return

    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0

    y = np.array([theta1_prev, omega1_prev, theta2_prev, omega2_prev])

    def ODE_system(state):
        theta1, omega1, theta2, omega2 = state
        delta = theta1 - theta2
        den = 2 * mass1 + mass2 - mass2 * np.cos(2 * delta)
        dtheta1_dt = omega1
        dtheta2_dt = omega2
        domega1_dt = (-gravity * (2 * mass1 + mass2) * np.sin(theta1)
                      - mass2 * gravity * np.sin(theta1 - 2 * theta2)
                      - 2 * np.sin(delta) * mass2 * (omega2 ** 2 * length2 + omega1 ** 2 * length1 * np.cos(delta))) / (length1 * den)
        domega2_dt = (2 * np.sin(delta) * (omega1 ** 2 * length1 * (mass1 + mass2)
                     + gravity * (mass1 + mass2) * np.cos(theta1)
                     + omega2 ** 2 * length2 * mass2 * np.cos(delta))) / (length2 * den)
        return np.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

    k1 = dt * ODE_system(y)
    k2 = dt * ODE_system(y + a21 * k1)
    k3 = dt * ODE_system(y + a31 * k1 + a32 * k2)
    k4 = dt * ODE_system(y + a41 * k1 + a42 * k2 + a43 * k3)
    k5 = dt * ODE_system(y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
    k6 = dt * ODE_system(y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
    k7 = dt * ODE_system(y + a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)

    y_new = y + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7

    theta1_new, omega1_new, theta2_new, omega2_new = y_new

    angles1.append(theta1_new)
    velocities1.append(omega1_new)
    angles2.append(theta2_new)
    velocities2.append(omega2_new)

    x1, y1, x2, y2 = coordinates(theta1_new, theta2_new)
    KE1_new = 0.5 * mass1 * (length1 * omega1_new) ** 2
    KE2_new = 0.5 * mass2 * ((length1 * omega1_new) ** 2 + (length2 * omega2_new) ** 2 + 2 * length1 * length2 * omega1_new * omega2_new * np.cos(theta1_new - theta2_new))
    PE1_new = mass1 * gravity * y1
    PE2_new = mass2 * gravity * y2
    KE.append(KE1_new + KE2_new)
    PE.append(PE1_new + PE2_new)
    ME.append(KE1_new + KE2_new + PE1_new + PE2_new)
    error.append(KE1_new + KE2_new + PE1_new + PE2_new - actualME)

    return simulate(index + 1, theta1_new, omega1_new, theta2_new, omega2_new)

simulate(1, theta1_0, omega1_0, theta2_0, omega2_0)

fig, ax = plt.subplots()
ax.set_xlim(-(length1 + length2) * 1.2, (length1 + length2) * 1.2)
ax.set_ylim(-(length1 + length2) * 1.2, (length1 + length2) * 1.2)
ax.set_aspect('equal')

line1, = ax.plot([], [], lw=3, c='k')
line2, = ax.plot([], [], lw=3, c='k')
bob1 = plt.Circle((0, 0), 0.05, fc='b')
bob2 = plt.Circle((0, 0), 0.05, fc='r')
ax.add_patch(bob1)
ax.add_patch(bob2)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    bob1.set_center((0, 0))
    bob2.set_center((0, 0))
    return line1, line2, bob1, bob2

kinetic_text = ax.text(-(length1 + length2), (length1 + length2) * 1.3, '', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
potential_text = ax.text(-(length1 + length2), (length1 + length2) * 1.1, '', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
mechanical_text = ax.text(-(length1 + length2), (length1 + length2) * 0.9, '', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
ME_error = ax.text(-(length1 + length2), (length1 + length2) * 0.7, '', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

def update(frame):
    x1, y1, x2, y2 = coordinates(angles1[frame], angles2[frame])
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    bob1.set_center((x1, y1))
    bob2.set_center((x2, y2))
    kinetic_text.set_text(f'Kinetic Energy: {KE[frame]:.2f} J')
    potential_text.set_text(f'Potential Energy: {PE[frame]:.2f} J')
    mechanical_text.set_text(f'Total Energy: {ME[frame]:.2f} J')
    ME_error.set_text(f'ME Error: {error[frame]:.2f} J')
    return line1, line2, bob1, bob2, kinetic_text, potential_text, mechanical_text, ME_error

ani = FuncAnimation(fig, update, frames=int(steps), init_func=init, blit=True, interval=dt*1000)

plt.show()
