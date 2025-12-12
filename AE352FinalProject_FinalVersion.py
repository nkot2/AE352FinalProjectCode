import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

m = 1.5
g = 9.81
l = 0.2
R_disk = 0.127

k_thrust = 1.2e-5 #realistic value for a drone
b_drag = 1.5e-7 #realistic value for a drone

Ixx = 0.5*m*R_disk**2 + 2*m*l**2
Iyy = Ixx
Izz = 0.5*m*R_disk**2 + 4*m*l**2
I = np.diag([Ixx, Iyy, Izz])

def R_body_to_inertial(phi, theta, psi):
    R = np.array([
        [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
        [-np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi),
         np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi),
         np.sin(phi)*np.cos(theta)],
        [np.sin(phi)*np.sin(psi) + np.cos(phi)*np.sin(theta)*np.cos(psi),
         -np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi),
         np.cos(phi)*np.cos(theta)]
    ])
    return R

def euler_rates_from_body_rates(phi, theta, p, q, r):
    T = np.array([
        [1, np.tan(theta)*np.sin(phi), np.tan(theta)*np.cos(phi)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    return T @ np.array([p, q, r])

def allocate_motors_from_forces(T_des, tau_phi, tau_theta, tau_psi):
    A = np.array([
        [k_thrust, k_thrust, k_thrust, k_thrust],
        [0, -l*k_thrust, 0, l*k_thrust],
        [-l*k_thrust, 0, l*k_thrust, 0],
        [-b_drag, b_drag, -b_drag, b_drag]
    ])
    b = np.array([T_des, tau_phi, tau_theta, tau_psi])
    w_sq, _ = nnls(A, b)
    return w_sq

def dynamics(state, w_sq):
    x, y, z, phi, theta, psi, u, v, w, p, q, r = state
    T = k_thrust * np.sum(w_sq)
    tau_phi = l * k_thrust * (w_sq[3] - w_sq[1])
    tau_theta = l * k_thrust * (w_sq[2] - w_sq[0])
    tau_psi = b_drag * (w_sq[1] + w_sq[3] - w_sq[0] - w_sq[2])
    Rbi = R_body_to_inertial(phi, theta, psi)
    Rib = Rbi.T
    Gb = Rib @ np.array([0.0, 0.0, -m*g])
    Fb = Gb + np.array([0.0, 0.0, T])
    du = Fb[0]/m - (q*w - r*v)
    dv = Fb[1]/m - (r*u - p*w)
    dw = Fb[2]/m - (p*v - q*u)
    omega = np.array([p,q,r])
    Tau = np.array([tau_phi, tau_theta, tau_psi])
    H = I @ omega
    omega_cross_H = np.cross(omega, H)
    pqr_dot = np.linalg.inv(I) @ (Tau - omega_cross_H)
    dp, dq, dr = pqr_dot
    V_I = Rbi @ np.array([u,v,w])
    dx, dy, dz = V_I
    phi_dot, theta_dot, psi_dot = euler_rates_from_body_rates(phi, theta, p, q, r)
    return np.array([dx,dy,dz, phi_dot,theta_dot,psi_dot, du,dv,dw, dp,dq,dr])

params = {
    "altitude": (1, 2),
    "phi": (40.0, 8.0),
    "theta": (40.0, 8.0),
    "psi": (20.0, 5.0),
}

def control(state, desired, params):
    x,y,z,phi,theta,psi,u,v,w,p,q,r = state
    Kp_z, Kd_z = params["altitude"]
    Kp_phi,Kd_phi = params["phi"]
    Kp_theta,Kd_theta = params["theta"]
    Kp_psi,Kd_psi = params["psi"]
    z_err = desired["z_ref"] - z
    T_des = m*(g + Kp_z*z_err - Kd_z*w)
    T_des = max(T_des, 0.0)
    phi_err = desired["phi_ref"] - phi
    theta_err = desired["theta_ref"] - theta
    psi_err = (desired["psi_ref"] - psi + np.pi)%(2*np.pi) - np.pi
    tau_phi = Kp_phi * phi_err - Kd_phi * p
    tau_theta = Kp_theta*theta_err - Kd_theta*q
    tau_psi = Kp_psi * psi_err - Kd_psi * r
    return allocate_motors_from_forces(T_des,tau_phi,tau_theta,tau_psi)

def rk4_step(state, dt, w_sq):
    k1 = dynamics(state,w_sq)
    k2 = dynamics(state+0.5*dt*k1, w_sq)
    k3 = dynamics(state+0.5*dt*k2, w_sq)
    k4 = dynamics(state+dt*k3, w_sq)
    return state + dt/6*(k1+2*k2+2*k3+k4)

def run_mission(sim_steps, dt, state, controller):
    logs = {k: [] for k in ["t","x","y","z","phi","theta","psi"]}
    for i in range(sim_steps):
        t = i*dt
        desired = controller(t, state)
        w_sq = control(state, desired, params)
        state = rk4_step(state, dt, w_sq)
        logs["t"].append(t)
        logs["x"].append(state[0])
        logs["y"].append(state[1])
        logs["z"].append(state[2])
        logs["phi"].append(state[3])
        logs["theta"].append(state[4])
        logs["psi"].append(state[5])
    return logs

def ref_traj(t):
    t_asc = 6
    t_straight1 = 12
    t_hover1 = 3
    t_yaw = 6
    t_straight2 = 12
    t_hover2 = 3
    t_land = 100
    T1 = t_asc
    T2 = T1+t_straight1
    T3 = T2+t_hover1
    T4 = T3+t_yaw
    T5 = T4+t_straight2
    T6 = T5+t_hover2
    T7 = T6+t_land
    x=y=z=psi=0
    vx=vy=vz=0
    if t < T1:
        z = t/T1
        vz = 1/T1
    elif t < T2:
        dt=t-T1
        z=1
        x = (dt/t_straight1)*5
        vx = 5/t_straight1
    elif t < T3:
        x=5
        z=1
    elif t < T4:
        x=5
        z=1
        psi = (t-T3)/t_yaw * np.pi/2
    elif t < T5:
        x=5
        z=1
        psi=np.pi/2
        y = ((t-T4)/t_straight2)*5
        vy = 5/t_straight2
    elif t < T6:
        x=y=5
        z=1
        psi=np.pi/2
    elif t < T7:
        x=y=5
        psi=np.pi/2
        dt=t-T6
        z = 1 - dt/t_land
        vz = -1/t_land
    else:
        x=y=5
        psi=np.pi/2
        z=0
    return x,y,z,psi,vx,vy,vz

dt = 0.005
state1 = np.zeros(12)
state1[2] = 1.0
def mission1(t, s):
    return {"z_ref":1, "phi_ref":0, "theta_ref":0, "psi_ref":0}
log1 = run_mission(int(120/dt), dt, state1, mission1)

R = 2
v=0.5
omega=v/R
def mission2(t, state):
    x_ref = R*np.cos(omega*t)
    y_ref = R*np.sin(omega*t)
    x_dot = -R*omega*np.sin(omega*t)
    y_dot = R*omega*np.cos(omega*t)
    x_err = x_ref - state[0]
    y_err = y_ref - state[1]
    phi,theta,psi = state[3:6]
    v_world = R_body_to_inertial(phi,theta,psi) @ state[6:9]
    vx,vy,_ = v_world
    ax = 1.0*x_err + 0.5*(x_dot - vx)
    ay = 1.0*y_err + 0.5*(y_dot - vy)
    theta_ref = -ax/g
    phi_ref = ay/g
    limit=np.deg2rad(20)
    phi_ref=np.clip(phi_ref,-limit,limit)
    theta_ref=np.clip(theta_ref,-limit,limit)
    return {"z_ref":1,"phi_ref":phi_ref,"theta_ref":theta_ref,"psi_ref":0}

state2 = np.zeros(12)
state2[2] = 1
state2[0] = R
log2 = run_mission(int(60/dt), dt, state2, mission2)

def mission3(t, state):
    x_ref,y_ref,z_ref,psi_ref,vx_ref,vy_ref,vz_ref = ref_traj(t)
    x_err = x_ref - state[0]
    y_err = y_ref - state[1]
    phi,theta,psi = state[3:6]
    v_world = R_body_to_inertial(phi,theta,psi) @ state[6:9]
    vx,vy,_ = v_world
    ax = 1.0*x_err + 0.5*(vx_ref - vx)
    ay = 1.0*y_err + 0.5*(vy_ref - vy)
    theta_ref = -ax/g
    phi_ref = ay/g
    limit=np.deg2rad(20)
    phi_ref=np.clip(phi_ref,-limit,limit)
    theta_ref=np.clip(theta_ref,-limit,limit)
    return {"z_ref":z_ref,"phi_ref":phi_ref,"theta_ref":theta_ref,"psi_ref":psi_ref}

state3 = np.zeros(12)
total_T = 150
log3 = run_mission(int(total_T/dt), dt, state3, mission3)

# Goal 1
plt.figure(figsize=(6,4))
plt.plot(log1["t"], log1["z"], label="z")
plt.plot(log1["t"], log1["x"], label="x")
plt.plot(log1["t"], log1["y"], label="y")
plt.title("Mission 1: Hover – Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log1["t"], log1["phi"], label="phi")
plt.plot(log1["t"], log1["theta"], label="theta")
plt.plot(log1["t"], log1["psi"], label="psi")
plt.title("Mission 1: Hover – Euler Angles")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()

# Goal 2
plt.figure(figsize=(6,4))
plt.plot(log2["t"], log2["z"])
plt.title("Mission 2: Circle – Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log2["x"], log2["y"])
plt.title("Mission 2: Circle – XY Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log2["t"], log2["phi"], label="phi")
plt.plot(log2["t"], log2["theta"], label="theta")
plt.plot(log2["t"], log2["psi"], label="psi")
plt.title("Mission 2: Circle – Euler Angles")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()

# Goal 3
plt.figure(figsize=(6,4))
plt.plot(log3["t"], log3["z"])
plt.title("Mission 3: Full Path – Altitude")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log3["x"], log3["y"])
plt.title("Mission 3: Full Path – XY Path")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(log3["t"], log3["phi"], label="phi")
plt.plot(log3["t"], log3["theta"], label="theta")
plt.plot(log3["t"], log3["psi"], label="psi")
plt.title("Mission 3: Full Path – Euler Angles")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()
