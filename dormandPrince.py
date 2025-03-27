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

def simulate(index, theta_prev, velocity_prev):
    if index >= steps:
        return 
     
    #Dormand-Prince coefficients from the Butcher tableau
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
        
    #5th order solution coefficients
    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    
    #state vector [theta, omega]
    y = np.array([theta_prev, velocity_prev])
    
    #ordinary differential equations system
    def ODE_system(state):
        theta, omega = state
        dtheta_dt = omega
        domega_dt = -gravity / length * np.sin(theta)
        return np.array([dtheta_dt, domega_dt])
    
    #calculation for k values (slopes)
    t = index * dt
    k1 = dt * ODE_system(y)
    k2 = dt * ODE_system(y + a21*k1)
    k3 = dt * ODE_system(y + a31*k1 + a32*k2)
    k4 = dt * ODE_system(y + a41*k1 + a42*k2 + a43*k3)
    k5 = dt * ODE_system(y + a51*k1 + a52*k2 + a53*k3 + a54*k4)
    k6 = dt * ODE_system(y + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
    k7 = dt * ODE_system(y + a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6)
    
    #5th order solutions stored in y_new array
    y_new = y + b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7
    
    theta_new, velocity_new = y_new
    
    angles.append(theta_new)
    velocities.append(velocity_new)
    
    KE_new = 0.5 * mass * (length * velocity_new)**2
    PE_new = mass * gravity * coordinates(theta_new)[1]

    KE_new = 0.5 * mass * (length * velocity_new)**2
    PE_new = mass * gravity * coordinates(theta_new)[1]

    KE.append(KE_new)
    PE.append(PE_new)
    ME.append(KE_new + PE_new)
    error.append(PE_new + KE_new - actualME)

    return(simulate(index + 1, theta_new, velocity_new))

simulate(1, theta0, omega0)


#visual using matplotlib
fig, ax = plt.subplots()
ax.set_xlim(-length * 1.2, length * 1.2)
ax.set_ylim(-length * 1.2, length * 1.2)
ax.set_aspect('equal')

#add shapes
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