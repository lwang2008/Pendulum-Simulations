import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation

sys.setrecursionlimit(10000) 

#set up simulation variables
length=1
mass=1000
gravity = 100
dt=0.01
final_time = 10
steps = final_time/dt

#set up initial conditions of pendulum
theta0 = np.radians(60)
omega0 = 0
period = 2 * np.pi * np.sqrt(length / gravity)

#forces
gravityF = mass * gravity
radialF = 0

time_points = np.arange(0, final_time, dt)

def coordinates(theta):
    return length * np.sin(theta), -length * np.cos(theta)

#arrays to store results
angles = [theta0]
velocities = [omega0]
ypos = np.zeros(int(steps) +2)
KE = [0.5 * mass * (length * omega0)**2]
PE = [mass * gravity * coordinates(theta0)[1]]
actualME = KE[0] + PE[0]
ME = [KE[0] + PE[0]]
error = [ME[0] - actualME]


#calculuate new angular position and angular velocity using RK4
def rk4_step(theta, omega, dt):
    def angular_accel(theta):
        return -gravity / length *np.sin(theta)
    
    #calculuate intermediate slopes k1, k2, k3, k4
    k1_theta = omega
    k1_omega = angular_accel(theta)
    
    k2_theta = omega + 0.5 * k1_omega * dt
    k2_omega = angular_accel(theta + 0.5 * k1_theta * dt)
    
    k3_theta = omega + 0.5 * k2_omega * dt
    k3_omega = angular_accel(theta + 0.5 * k2_theta * dt)
    
    k4_theta = omega + k3_omega * dt
    k4_omega = angular_accel(theta + k3_theta * dt)

    theta_new = theta + (dt / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega_new = omega + (dt / 6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    
    return theta_new, omega_new


def simulate_rk4(index, theta_prev, velocity_prev):
    if index >= steps:
        return
    
    #use RK4 helper function to calculate next values
    theta_new, omega_new = rk4_step(theta_prev, velocity_prev, dt)
    
    KE_new = 0.5 * mass * (length * omega_new)**2
    PE_new = mass * gravity * coordinates(theta_new)[1]
    
    angles.append(theta_new)
    velocities.append(omega_new)
    KE.append(KE_new)
    PE.append(PE_new)
    ME.append(KE_new + PE_new)
    error.append(ME[-1] - actualME)
    
    return simulate_rk4(index + 1, theta_new, omega_new)

simulate_rk4(1, theta0, omega0)


#visual using matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-length * 1.2, length * 1.2)
ax.set_ylim(-length * 1.2, length * 1.2)
ax.set_aspect('equal')

#Add shapes
line, = ax.plot([], [], lw=3, c='k')  
bob = plt.Circle((0, 0), 0.05, fc='r')  
ax.add_patch(bob)


#initialize animation
def init():
    line.set_data([], [])
    bob.set_center((0, 0))
    return line, bob

#display kinetic, potential, mechanical energy and total mechanical energy error
kinetic_text = ax.text(-length, length * 1.3, '', fontsize=12, 
                        bbox=dict(facecolor='red', alpha=0.5))
potential_text = ax.text(-length, length * 1.1, '', fontsize=12,
                        bbox=dict(facecolor='blue', alpha=0.5))
mechanical_text = ax.text(-length, length * 0.9, '', fontsize=12,
                        bbox=dict(facecolor='yellow', alpha=0.5))
ME_error = ax.text(-length, length * 0.7, '', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.5))

#update each frame based on index 
def update(frame):
    x, y = coordinates(angles[frame])
    line.set_data([0, x], [0, y])
    bob.set_center((x, y))
    kinetic_text.set_text(f'Kinetic Energy: {KE[frame]:.2f} J')
    potential_text.set_text(f'Potential Energy: {PE[frame]:.2f} J')
    mechanical_text.set_text(f'Total Energy: {ME[frame]:.2f} J')
    ME_error.set_text(f'ME Error: {error[frame]:.2f} J')
    return line, bob, kinetic_text, potential_text, mechanical_text, ME_error


ani = FuncAnimation(fig, update, frames=int(steps), init_func=init, blit=True, interval=dt*1000)
plt.show()