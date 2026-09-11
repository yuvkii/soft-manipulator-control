import numpy as np
import math
from simulator_setup import setup_simulator
from plotting import plot_results

setup_data = setup_simulator()
sim = setup_data['sim']
plant = setup_data['plant']
plant_context = setup_data['plant_context']
j_motor1 = setup_data['j_motor1']
j_motor2 = setup_data['j_motor2']
j_spring1 = setup_data['j_spring1']
j_spring2 = setup_data['j_spring2']
j_wall = setup_data['j_wall']
L1 = setup_data['L1']
L2 = setup_data['L2']
phi1_init = setup_data['phi1_init']
phi2_init = setup_data['phi2_init']
k1 = setup_data['k1']
k2 = setup_data['k2']
wall_x = setup_data['wall_x']
wall_thickness = setup_data['wall_thickness']
wall_k = setup_data['wall_k']
wall_damping_extra = setup_data['wall_damping_extra']
wall_fric_normal = setup_data['wall_fric_normal']
wall_mu_static = setup_data['wall_mu_static']
wall_mu_coulomb = setup_data['wall_mu_coulomb']
wall_v_stiction = setup_data['wall_v_stiction']
TORQUE_LIMIT = setup_data['TORQUE_LIMIT']
USE_FALSE_TRAJ = setup_data['USE_FALSE_TRAJ']
PRINT_SUMMARY = setup_data['PRINT_SUMMARY']

jl1 = j_motor1
jl2 = j_motor2
jdel1 = j_spring1
jdel2 = j_spring2

# ========================
# CONTROL PARAMETERS 
# ========================
Kp = np.diag([80.0, 80.0])
Kd = np.diag([20.0, 20.0])

# ========================
# PHYSICAL PARAMETERS FOR DYNAMICS MODEL
# ========================
# Link masses (from simulator_setup)
m1_dyn = 1.5
m2_dyn = 1.5
g_acc  = 9.81


J1_dyn = (2.0 / 5.0) * 0.5 * 0.08**2
J2_dyn = (2.0 / 5.0) * 0.4 * 0.08**2

# ========================
# COMPUTED TORQUE GAINS
# ========================
Kp_ct = np.diag([100.0, 100.0])
Kd_ct = np.diag([20.0, 20.0])

# ========================
# FORWARD KINEMATICS
# ========================
def forward_kinematics(q1, q2):
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    z = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, z])

# ========================
# INVERSE KINEMATICS
# ========================
def inverse_kinematics(x, z):
    D = (x**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)

    q2 = np.arccos(D)
    q1 = np.arctan2(z, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))

    return np.array([q1, q2])

# ========================
# TRAJECTORY PARAMETERS
# ========================
TRAJ_X_WALL  = 1.088   
TRAJ_Z_CTR   = 0.40    
TRAJ_AMP     = 0.40    
TRAJ_FREQ    = 0.50    

# ========================
# TRAJECTORY 
# ========================
def trajectory(t):
    x = TRAJ_X_WALL
    z = TRAJ_Z_CTR + TRAJ_AMP * np.sin(2 * np.pi * TRAJ_FREQ * t)
    return np.array([x, z])

def trajectory_velocity(t):
    """Cartesian velocity of the desired trajectory [vx, vz]."""
    vx = 0.0
    vz = TRAJ_AMP * 2 * math.pi * TRAJ_FREQ * math.cos(2 * math.pi * TRAJ_FREQ * t)
    return np.array([vx, vz])

def trajectory_acceleration(t):
    """Cartesian acceleration of the desired trajectory [ax, az]."""
    ax = 0.0
    az = -TRAJ_AMP * (2 * math.pi * TRAJ_FREQ)**2 * math.sin(2 * math.pi * TRAJ_FREQ * t)
    return np.array([ax, az])

# ========================
# JACOBIAN
# ========================
def jacobian(q):
    """Geometric Jacobian for the 2-DOF planar arm (x-z plane)."""
    q1  = q[0]
    q12 = q[0] + q[1]
    J = np.array([
        [-L1 * np.sin(q1) - L2 * np.sin(q12),  -L2 * np.sin(q12)],
        [ L1 * np.cos(q1) + L2 * np.cos(q12),   L2 * np.cos(q12)]
    ])
    return J

def jacobian_dot_qdot(q, qd):
    """Computes J_dot * qd analytically (used for feedforward acceleration)."""
    q1  = q[0]
    q12 = q[0] + q[1]
    c1  = math.cos(q1)
    s1  = math.sin(q1)
    c12 = math.cos(q12)
    s12 = math.sin(q12)
    qd1, qd2 = qd[0], qd[1]
    qd12 = qd1 + qd2

    
    dJ11_dt = -L1 * c1 * qd1 - L2 * c12 * qd12
    dJ12_dt = -L2 * c12 * qd12
    dJ21_dt = -L1 * s1 * qd1 - L2 * s12 * qd12
    dJ22_dt = -L2 * s12 * qd12

    return np.array([
        dJ11_dt * qd1 + dJ12_dt * qd2,
        dJ21_dt * qd1 + dJ22_dt * qd2
    ])

# ========================
# DYNAMICS MATRICES
# ========================
def inertia_matrix(q):
    """M(q): 2x2 inertia matrix."""
    c2  = math.cos(q[1])
    M11 = (m1_dyn + m2_dyn) * L1**2 + m2_dyn * L2**2 + 2 * m2_dyn * L1 * L2 * c2
    M12 = m2_dyn * L2**2 + m2_dyn * L1 * L2 * c2
    M22 = m2_dyn * L2**2
    return np.array([[M11, M12],
                     [M12, M22]])

def coriolis_matrix(q, qd):
    """C(q, qd): 2x2 Coriolis/centrifugal matrix."""
    h = m2_dyn * L1 * L2 * math.sin(q[1])
    return np.array([[-2 * h * qd[1], -h * qd[1]],
                     [ h * qd[0],      0.0        ]])

def gravity_torque(q):
    """tau_g(q): gravitational torque vector."""
    q1  = q[0]
    q12 = q[0] + q[1]
    tg1 = ((m1_dyn + m2_dyn) * g_acc * L1 * math.sin(q1)
            + m2_dyn * g_acc * L2 * math.sin(q12))
    tg2 = m2_dyn * g_acc * L2 * math.sin(q12)
    return np.array([tg1, tg2])

def motor_inertia():
    """J: 2x2 diagonal motor inertia matrix."""
    return np.diag([J1_dyn, J2_dyn])

# ========================
# COMPUTED TORQUE CONTROLLER
# ========================
def computed_torque_control(q, qd, q_des, qd_des, qdd_des):
    M_eff = inertia_matrix(q) + motor_inertia()
    C     = coriolis_matrix(q, qd)
    g     = gravity_torque(q)

    e    = q_des  - q
    edot = qd_des - qd

    v   = qdd_des + Kd_ct @ edot + Kp_ct @ e
    tau = M_eff @ v + C @ qd + g

    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

# ========================
# CONTROLLER 
# ========================
def controller(q, qd, q_des, qd_des):
    e = q_des - q
    edot = qd_des - qd

    tau = Kp @ e + Kd @ edot

    tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
    return tau

# ============================================
# SIMULATION LOOP WITH COMPLETE CONTROL
# ============================================

print("\n" + "="*60)
print("RUNNING SIMULATION")
print("="*60)

time_log = []
q_log = []
q_des_log = []
qd_log = []
qd_des_log = []
qdd_log = []
qdd_des_log = []
delta_log = []
delta_des_log = []
theta_log = []
theta_des_log = []
thetadot_log = []
thetadot_des_log = []
ee_log = []
ee_des_log = []
tau_log = []
tau_link_log = []
contact_log = []

sim.set_target_realtime_rate(1.0)
dt_vis = 0.02
t = 0.0
T_final = 30.0

while t < T_final:

    #  ----Acquire Feedback, Please don't modify this section----
    theta = np.array([jl1.get_angle(plant_context), jl2.get_angle(plant_context)])
    thetadot = np.array([jl1.get_angular_rate(plant_context), jl2.get_angular_rate(plant_context)])

    spring_delta = np.array([jdel1.get_angle(plant_context), jdel2.get_angle(plant_context)])
    spring_deltad = np.array([jdel1.get_angular_rate(plant_context), jdel2.get_angular_rate(plant_context)])
    wall_pos = j_wall.get_translation(plant_context)
    wall_vel = j_wall.get_translation_rate(plant_context)

    wall_drive = -wall_k * wall_pos - wall_damping_extra * wall_vel
    normal_force = max(wall_fric_normal, wall_k * wall_pos)
    fric_static = wall_mu_static * normal_force
    fric_coulomb = wall_mu_coulomb * normal_force
    if abs(wall_vel) < wall_v_stiction and abs(wall_drive) < fric_static:
        wall_fric = -wall_drive
    else:
        wall_fric = -fric_coulomb * np.sign(wall_vel)

    wall_force = wall_drive + wall_fric

    q  = theta + spring_delta
    qd = thetadot + spring_deltad

   
    ee_current = forward_kinematics(q[0], q[1])

    p_des = trajectory(t)

  
    q_des = inverse_kinematics(p_des[0], p_des[1])


    J_mat  = jacobian(q_des)
    xd_des = trajectory_velocity(t)
    try:
        qd_des = np.linalg.solve(J_mat, xd_des)
    except np.linalg.LinAlgError:
        qd_des = np.zeros(2)

    xdd_des    = trajectory_acceleration(t)
    Jdot_qdot  = jacobian_dot_qdot(q_des, qd_des)
    try:
        qdd_des = np.linalg.solve(J_mat, xdd_des - Jdot_qdot)
    except np.linalg.LinAlgError:
        qdd_des = np.zeros(2)

   
    tau = computed_torque_control(q, qd, q_des, qd_des, qdd_des)

    tau_link = np.array([k1 * spring_delta[0], k2 * spring_delta[1]])

 
    M_des   = inertia_matrix(q_des)
    C_des   = coriolis_matrix(q_des, qd_des)
    tg_des  = gravity_torque(q_des)
    lhs_des = -(M_des @ qdd_des + C_des @ qd_des + tg_des)
    delta_des = np.array([lhs_des[0] / k1, lhs_des[1] / k2])

    theta_des    = q_des - delta_des
    thetadot_des = qd_des   

    # -----Please don't modify the wall_force in here ------
    u = np.array([tau[0], tau[1], wall_force])
    plant.get_actuation_input_port().FixValue(plant_context, u)
    sim.AdvanceTo(t)
    t += dt_vis

    # -----Logging data, please don't modify this section ------
    q_wall_pos = j_wall.get_translation(plant_context)
    wall_surface_x_nominal = (wall_x + q_wall_pos) - wall_thickness / 2.0
    dist_to_wall = wall_surface_x_nominal - ee_current[0]

    if dist_to_wall <= 0.016:
         contact_log.append(ee_current.copy())

    time_log.append(t)
    q_log.append(q.copy())
    q_des_log.append(q_des.copy())
    qdd_actual = (qd - qd_log[-1]) / dt_vis if qd_log else np.zeros(2)
    qd_log.append(qd.copy())
    qd_des_log.append(qd_des.copy())
    qdd_log.append(qdd_actual)
    qdd_des_log.append(qdd_des.copy())
    delta_log.append(spring_delta.copy())
    delta_des_log.append(delta_des.copy())
    theta_log.append(theta.copy())
    theta_des_log.append(theta_des.copy())
    thetadot_log.append(thetadot.copy())
    thetadot_des_log.append(thetadot_des.copy())
    ee_log.append(ee_current.copy())
    ee_des_log.append(p_des.copy())
    tau_log.append(tau.copy())
    tau_link_log.append(tau_link.copy())

q_final = np.array(q_log[-1])
ee_final = forward_kinematics(q_final[0], q_final[1])
if ee_final is None:
    ee_final = np.array([0.0, 0.0])

if PRINT_SUMMARY:
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("Final joint angles  : phi1={:.4f} rad, phi2={:.4f} rad".format(q_final[0], q_final[1]))
    print("Final EE position   : x={:.4f} m, z={:.4f} m".format(ee_final[0], ee_final[1]))
    des_final = np.array(ee_des_log[-1])
    print("Target EE position  : x={:.4f} m, z={:.4f} m".format(des_final[0], des_final[1]))
    print("Position error      : {:.4f} m".format(np.linalg.norm(ee_final - des_final)))

plot_results(time_log, q_log, q_des_log, qd_log, qd_des_log, qdd_log, qdd_des_log,
             delta_log, delta_des_log, theta_log, theta_des_log, thetadot_log,
             thetadot_des_log, ee_log, ee_des_log, tau_log, tau_link_log, contact_log, PRINT_SUMMARY)

print("\n" + "="*60)
print("MeshCat visualization is running...")
print("="*60)
input("\nPress Enter to quit...")
