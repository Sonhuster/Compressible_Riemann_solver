import numpy as np
from input import *

# --------------------------Initialization------------------------------ #
class Fluid:
    def __init__(self, mesh, init_u, init_v, init_p, init_T, init_rho):
        n_cells = mesh.no_elems()
        n_faces = mesh.no_faces()
        n_nodes = mesh.no_nodes()

        self.uc = np.zeros(n_cells) + init_u
        self.vc = np.zeros(n_cells) + init_v
        self.pc = np.zeros(n_cells) + init_p
        self.Tc = np.zeros(n_cells) + init_T
        self.rhoc = np.zeros(n_cells) + init_rho

        self.uf = np.zeros(n_faces)
        self.vf = np.zeros(n_faces)
        self.pf = np.zeros(n_faces)
        self.Tf = np.zeros(n_faces)
        self.rhof = np.zeros(n_faces)

        self.uv = np.zeros(n_nodes)
        self.vv = np.zeros(n_nodes)
        self.pv = np.zeros(n_nodes)
        self.Tv = np.zeros(n_nodes)
        self.rhov = np.zeros(n_nodes)

        self.ubc = np.zeros(len(mesh.boundary_info.faces))
        self.vbc = np.zeros(len(mesh.boundary_info.faces))
        self.pbc = np.zeros(len(mesh.boundary_info.faces))
        self.Tbc = np.zeros(len(mesh.boundary_info.faces))
        self.rhobc = np.zeros(len(mesh.boundary_info.faces))


def init_LR_state(mesh):
    n_faces = mesh.no_faces()

    # rhoR, uR, vR, pR, hR, eR, TR
    state_L = np.zeros((7, n_faces))
    state_R = np.zeros((7, n_faces))
    return state_L, state_R