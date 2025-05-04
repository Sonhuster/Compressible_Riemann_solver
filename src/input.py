# Simulation mode
SIMULATION_MODE = 2     # 0: Lid driven cavity | 1: airfoil flow | 2: channel flow | 3: Heat conduction

# Convective scheme
convection_scheme = 0          # ROE's scheme | AUSM's scheme

# Mesh
READ_MESH = False
inflow_length = 0.4     # For airflow case

# Input definition
tol_outer = 1e-06
iter_outer = 2000

# Time derivative
dt = 0.001
rk4 = [1/6, 1/3, 1/3, 1/6]

relax_uv = 0.8
relax_p = 0.1
noise = 0.005

# Flow properties
mu = 0.01
rho = 1.0
shr = 1.4
Pr = 0.72

u_inf = 1.0
v_inf = 0.0
p_inf = 1.0
T_inf = 1.0

# Post-Processing
n_plot = 10
streamline_mode = False