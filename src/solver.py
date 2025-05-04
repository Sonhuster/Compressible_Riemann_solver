import numpy as np
import config as cf
import bc
import utils as ufs
from input import *

"""
Symbol explanation:
    - In order to vectorize formulations, I used the combination symbols to mark array shape, it this function,
    they include three types:
        + _var  implies this array's head is shaped (number of elements) x (number of local faces).
        +  var_ implies this array's head is shaped (number of global faces) x (2).
        + _var_ implies this array's head is shaped (number of elements) x (2) x (number of local faces).
    The tail (number of elements along the latest axis) of the var array depends on its original tail, for example:
        + sn has tail == 2, so _sn = (number of elements) x (number of local faces) x (2)
        + area has tail == 1, so _area = (number of elements) x (number of local faces) x (1)
"""


def cal_node_face_value(mesh, var, fw, cw):
    var.uv = np.sum(var.uc * cw, axis=1)
    var.vv = np.sum(var.vc * cw, axis=1)
    var.pv = np.sum(var.pc * cw, axis=1)
    var.Tv = np.sum(var.Tc * cw, axis=1)

    # Cal boundary conditions (update var_bc, var_v, var_f <- from var_v (Neumann))
    var.uf = cal_face_value(mesh, fw, var.uc, var.uf)
    var.vf = cal_face_value(mesh, fw, var.vc, var.vf)
    var.pf = cal_face_value(mesh, fw, var.pc, var.pf)
    var.Tf = cal_face_value(mesh, fw, var.Tc, var.Tf)
    var = bc.set_bc(mesh, var, SIMULATION_MODE)

    return var

def cal_face_value(mesh, fw, var, varf, bc_face=False):
    f2c, f_2_bf = ufs.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    (var_,) = mesh.var_face_wise(var,)
    weighted_coeff = np.vstack((fw, 1 - fw)).T
    if not bc_face:
        maskf_bc = f_2_bf >= 0
        maskf_in = ~ maskf_bc

        # Keep face values on BCs
        varf_bc = varf * maskf_bc
        varf_in = np.sum(var_ * weighted_coeff, axis=1) * maskf_in

        return varf_bc + varf_in
    else:
        return np.sum(var_ * weighted_coeff, axis=1)

# First order
def cal_left_right_state(mesh, fw, var):
    f2c, f_2_bf = ufs.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    snsign,  = ufs.shorted_name(mesh.global_faces, 'snsign')
    weighted_coeff = np.vstack((fw, 1 - fw)).T
    mask_bc_dirichlet = bc.get_fix_vel_bc_mask(mesh, SIMULATION_MODE)
    mask_in_dirichlet = ~ mask_bc_dirichlet

    def cal_var_state(var: np.array, varbc: np.array):
        (var_, varbc_) = mesh.var_face_wise(var, varbc)

        # Keep face values on BCs
        varf_bc = varbc_ * mask_bc_dirichlet
        varf_in = var_ * mask_in_dirichlet.reshape(-1, 1)

        # varL = varf_in[:, 0] + varf_bc
        varL = var_[:, 0]
        varR = varf_in[:, 1] + varf_bc

        return varL, varR

    uL, uR = cal_var_state(var.uc, var.ubc)
    vL, vR = cal_var_state(var.vc, var.vbc)
    pL, pR = cal_var_state(var.pc, var.pbc)
    TL, TR = cal_var_state(var.Tc, var.Tbc)

    rhoL = shr * pL / TL
    rhoR = shr * pR / TR

    hL = TL / (shr - 1) + 0.5 * (uL * uL + vL * vL)
    hR = TR / (shr - 1) + 0.5 * (uR * uR + vR * vR)
    eL = pL / (shr - 1) + 0.5 * rhoL * (uL * uL + vL * vL)
    eR = pR / (shr - 1) + 0.5 * rhoR * (uR * uR + vR * vR)

    state_L = [rhoL, uL, vL, pL, hL, eL, TL]
    state_R = [rhoR, uR, vR, pR, hR, eR, TR]

    # (init_state_L, init_state_R) = mesh.var_elem_wise(state_L.T, state_R.T)
    # (corrected_state_L, corrected_state_R) = mesh.var_elem_wise(state_L.T, state_R.T)
    # filter_idx = np.where(snsign == -1)
    # corrected_state_L[filter_idx] = init_state_R[filter_idx]
    # corrected_state_R[filter_idx] = init_state_L[filter_idx]
    # corrected_state_L = corrected_state_L.transpose(2, 0, 1)
    # corrected_state_R = corrected_state_R.transpose(2, 0, 1)
    # L = [state for state in corrected_state_L]
    # R = [state for state in corrected_state_R]

    return state_L, state_R


def cal_face_grad_n(mesh, varc, varv, varbc):
    delta, st, snsign = ufs.shorted_name(mesh.global_faces, 'delta', 'st', 'snsign')
    c2f, f2c, f2v, f_2_bf = ufs.shorted_name(mesh.link, 'c2f', 'f2c', 'f2v', 'f_2_bf')
    centroid = ufs.shorted_name(mesh.elems, 'centroid')[0]

    mask_bc_dirichlet = bc.get_fix_vel_bc_mask(mesh, SIMULATION_MODE)
    mask_bc_neumann = bc.get_outflow_bc_mask(mesh, SIMULATION_MODE)
    mask_bc, mask_in = mesh.get_face_mask_element_wise()
    (_mask_bc_dirichlet, _mask_bc_neumann) = mesh.var_elem_wise(mask_bc_dirichlet, mask_bc_neumann)
    _mask_in_dirichlet, _mask_in_neumann = ~_mask_bc_dirichlet, ~_mask_bc_neumann
    (_delta, _st, _varv, _varbc, _f2c, _f2v) = mesh.var_elem_wise(delta, st, varv, varbc, f2c, f2v)
    icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])

    # Diagonal coefficient
    ap = (varc.reshape(-1, 1) / _delta) * _mask_in_neumann
    # Off-diagonal coefficient
    anb = (- varc[icn] / _delta) * mask_in
    # Source terms
    sc =  (- _varbc / _delta) * _mask_bc_dirichlet
    # Skew term
    d1 = centroid[_f2c][:, :, 1, :] - centroid[_f2c][:, :, 0, :]
    tdotl = np.sum(_st * d1, axis=2)
    skew = (tdotl * (varv[_f2v][:, :, 1] - varv[_f2v][:, :, 0]) * snsign / _delta) * _mask_bc_dirichlet

    if np.sum(skew) > 1e-05:
        raise "Skew term wrong"

    grad_f = ap + anb + sc - skew
    (grad_f, ) = ufs.get_total_flux(grad_f, )

    return grad_f


def AUSM_flux(mesh, state_L, state_R):
    area, delta, snsign, sn = ufs.shorted_name(mesh.global_faces, 'area', 'delta', 'snsign', 'sn')
    rhoL, uL, vL, pL, hL, TL = state_L[0], state_L[1], state_L[2], state_L[3], state_L[4], state_L[6]
    rhoR, uR, vR, pR, hR, TR = state_R[0], state_R[1], state_R[2], state_R[3], state_R[4], state_R[6]

    # cL = np.sqrt(TL)
    # cR = np.sqrt(TR)
    cL = uL
    cR = uR

    MaL = np.einsum('ij,ij->i', np.vstack((uL, vL)).T, sn) / cL
    MaR = np.einsum('ij,ij->i', np.vstack((uR, vR)).T, sn) / cR

    Mp = np.where(MaL <= -1.0, 0.0, np.where(MaL < 1.0, 0.25 * (MaL + 1.0) ** 2, MaL))
    pp = np.where(MaL <= -1.0, 0.0, np.where(MaL < 1.0, 0.25 * pL * (MaL + 1) ** 2 * (2 - MaL), pL))
    Mm = np.where(MaR <= -1.0, MaR, np.where(MaR < 1.0, -0.25 * (MaR - 1.0) ** 2, 0.0))
    pm = np.where(MaR <= -1.0, pR, np.where(MaR < 1.0, 0.25 * pR * (MaR - 1) ** 2 * (2 + MaR), 0.0))

    Mpm = Mp + Mm  # shape (N,)

    rho_flux  = 0.5 * Mpm * (rhoL * cL      + rhoR * cR)
    rhou_flux = 0.5 * Mpm * (rhoL * cL * uL + rhoR * cR * uR)
    rhov_flux = 0.5 * Mpm * (rhoL * cL * vL + rhoR * cR * vR)
    rhoe_flux = 0.5 * Mpm * (rhoL * cL * hL + rhoR * cR * hR)

    rho_flux  += 0.5 * np.abs(Mpm) * (rhoL * cL      - rhoR * cR)
    rhou_flux += 0.5 * np.abs(Mpm) * (rhoL * cL * uL - rhoR * cR * uR)
    rhov_flux += 0.5 * np.abs(Mpm) * (rhoL * cL * vL - rhoR * cR * vR)
    rhoe_flux += 0.5 * np.abs(Mpm) * (rhoL * cL * hL - rhoR * cR * hR)

    rho_flux  += 0
    rhou_flux += sn[:, 0] * (pp + pm)
    rhov_flux += sn[:, 1] * (pp + pm)
    rhoe_flux += 0

    conv_flux = [rho_flux, rhou_flux, rhov_flux, rhoe_flux]

    return conv_flux

def ROE_flux(mesh, state_L: list, state_R: list):
    area, delta, sn, snsign = ufs.shorted_name(mesh.global_faces, 'area', 'delta', 'sn', 'snsign')
    phiL = [state_L[0], state_L[0] * state_L[1], state_L[0] * state_L[2], state_L[0] * state_L[5]]
    phiR = [state_R[0], state_R[0] * state_R[1], state_R[0] * state_R[2], state_R[0] * state_R[5]]

    rL = state_L[0]
    uL = state_L[1]
    vL = state_L[2]
    unL = uL*sn[:, 0] + vL*sn[:, 1]
    qL = np.sqrt(phiL[1] ** 2 + phiL[2] ** 2) / rL
    pL = (shr-1)*(phiL[3] - 0.5*rL*qL**2)
    if (np.any(pL < 0)) | (np.any(rL < 0)):
        raise ValueError('Non-physical state!')
    rHL = phiL[3] + pL
    HL = rHL/rL

    # left flux
    FL = np.array([
        rL * unL,
        phiL[1] * unL + pL * sn[:, 0],
        phiL[2] * unL + pL * sn[:, 1],
        rHL * unL
    ])

    # process right state
    rR = state_R[0]
    uR = state_R[1] / rR
    vR = state_R[2] / rR
    unR = uR * sn[:, 0] + vR * sn[:, 1]
    qR = np.sqrt(phiR[1] ** 2 + phiR[2] ** 2) / rR
    pR = (shr - 1) * (phiR[3] - 0.5 * rR * qR ** 2)
    if np.any(pR < 0) or np.any(rR < 0):
        raise ValueError('Non-physical state!')
    rHR = phiR[3] + pR
    HR = rHR / rR

    # right flux
    FR = np.array([
        rR * unR,
        phiR[1] * unR + pR * sn[:, 0],
        phiR[2] * unR + pR * sn[:, 1],
        rHR * unR
    ])

    # difference in states
    du = np.array(phiR) - np.array(phiL)

    # Roe average
    di = np.sqrt(rR / rL)
    d1 = 1.0 / (1.0 + di)

    ui = (di * uR + uL) * d1
    vi = (di * vR + vL) * d1
    Hi = (di * HR + HL) * d1

    af = 0.5 * (ui * ui + vi * vi)
    ucp = ui * sn[:, 0] + vi * sn[:, 1]
    c2 = (shr - 1) * (Hi - af)
    if np.any(c2 < 0):
        raise ValueError('Non-physical state!')
    ci = np.sqrt(c2)
    ci1 = 1.0 / ci

    # eigenvalues
    l = np.array([
        ucp + ci,
        ucp - ci,
        ucp
    ])

    # entropy fix
    epsilon = ci * .1
    entropy_fix_mask = np.abs(l) < epsilon
    l = np.where(entropy_fix_mask, 0.5 * (epsilon + l ** 2 / epsilon), np.abs(l))

    l = np.abs(l)
    l3 = l[2]

    # average and half-difference of 1st and 2nd eigs
    s1 = 0.5 * (l[0] + l[1])
    s2 = 0.5 * (l[0] - l[1])

    # left eigenvector product generators (see Theory guide)
    G1 = (shr - 1) * (af * du[0] - ui * du[1] - vi * du[2] + du[3])
    G2 = -ucp * du[0] + du[1] * sn[:, 0] + du[2] * sn[:, 1]

    # required functions of G1 and G2 (again, see Theory guide)
    C1 = G1 * (s1 - l3) * ci1 * ci1 + G2 * s2 * ci1
    C2 = G1 * s2 * ci1 + G2 * (s1 - l3)

    # flux assembly
    F = [
        0.5 * (FL[0] + FR[0]) - 0.5 * (l3 * du[0] + C1),
        0.5 * (FL[1] + FR[1]) - 0.5 * (l3 * du[1] + C1 * ui + C2 * sn[:, 0]),
        0.5 * (FL[2] + FR[2]) - 0.5 * (l3 * du[2] + C1 * vi + C2 * sn[:, 1]),
        0.5 * (FL[3] + FR[3]) - 0.5 * (l3 * du[3] + C1 * Hi + C2 * ucp)
    ]

    return F


def diffusion_flux(mesh, var, state_L, state_R):
    sn, snsign, area, delta = ufs.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area', 'delta')
    _area, _sn, _uf, _vf, _delta = mesh.var_elem_wise(area, sn, var.uf, var.vf, delta)
    k_Th = mu / ((shr - 1) * u_inf ** 2 * Pr)

    uL, vL, TL = state_L[1], state_L[2], state_L[6]
    uR, vR, TR = state_R[1], state_R[2], state_R[6]

    u_grad_f = (uR - uL) / delta
    v_grad_f = (vR - vL) / delta
    T_grad_f = (TR - TL) / delta

    nx, ny = sn[:, 0], sn[:, 1]

    u_grad_fx = u_grad_f * nx
    v_grad_fx = v_grad_f * nx
    T_grad_fx = T_grad_f * nx
    u_grad_fy = u_grad_f * ny
    v_grad_fy = v_grad_f * ny
    T_grad_fy = T_grad_f * ny

    div_U = u_grad_fx + v_grad_fy

    tau_xx = mu * (2 * u_grad_fx - 2 / 3 * div_U)
    tau_yy = mu * (2 * v_grad_fy - 2 / 3 * div_U)
    tau_xy = tau_yx = mu * (u_grad_fy + v_grad_fx)

    rho_fx  = np.zeros(mesh.no_faces())
    rhou_fx = tau_xx
    rhov_fx = tau_xy
    rhoe_fx = tau_xx * var.uf + tau_xy * var.vf + k_Th * T_grad_fx

    rho_fy  = np.zeros(mesh.no_faces())
    rhou_fy = tau_yx
    rhov_fy = tau_yy
    rhoe_fy = tau_yx * var.uf + tau_yy * var.vf + k_Th * T_grad_fy

    rho_flux    = nx * rho_fx + ny * rho_fy
    rhou_flux   = nx * rhou_fx + ny * rhou_fy
    rhov_flux   = nx * rhov_fx + ny * rhov_fy
    rhoe_flux   = nx * rhoe_fx + ny * rhoe_fy

    diff_flux = [rho_flux, rhou_flux, rhov_flux, rhoe_flux]

    return diff_flux


def intergrate_flux(mesh, conv_flux, diff_flux):
    def f2c_flux(flux):
        (_flux, ) = mesh.var_elem_wise(flux, )
        _flux = _flux * mesh.global_faces.snsign
        return _flux

    area, = ufs.shorted_name(mesh.global_faces, 'area')
    vol,  = ufs.shorted_name(mesh.elems, 'volume')
    _area, = mesh.var_elem_wise(area)

    mask_in_neumann = ~ bc.get_outflow_bc_mask(mesh, SIMULATION_MODE)
    (_mask_in_neumann,) = mesh.var_elem_wise(mask_in_neumann, )

    conv_elems = []; diff_elems = []
    for conv, diff in zip(conv_flux, diff_flux):
        conv_elems.append(f2c_flux(conv))
        diff_elems.append(f2c_flux(diff) * _mask_in_neumann)

    rho_flux,  = ufs.get_total_flux(_area * (conv_elems[0] - diff_elems[0]), ) / vol
    rhou_flux, = ufs.get_total_flux(_area * (conv_elems[1] - diff_elems[1]), ) / vol
    rhov_flux, = ufs.get_total_flux(_area * (conv_elems[2] - diff_elems[2]), ) / vol
    rhoe_flux, = ufs.get_total_flux(_area * (conv_elems[3] - diff_elems[3]), ) / vol

    flux = [rho_flux, rhou_flux, rhov_flux, rhoe_flux]
    return flux


def cal_rk_temporal_var(var, flux, flux_rkt, rk):
    rho_temp  = - flux[0] * dt + var.rhoc
    rhou_temp = - flux[1] * dt + var.rhoc * var.uc
    rhov_temp = - flux[2] * dt + var.rhoc * var.vc
    e_temp    = - flux[3] * dt + var.rhoc * 0.5 * (var.uc**2 + var.vc**2) + var.pc / (shr - 1)

    var.rhoc = rho_temp
    var.uc   = rhou_temp / rho_temp
    var.vc   = rhov_temp / rho_temp
    var.pc   = (shr - 1) * (e_temp - 0.5 * (rhou_temp**2 + rhov_temp**2) / rho_temp)
    var.Tc   = shr * var.pc / rho_temp

    for cons_var in range(4):
        flux_rkt[cons_var] += rk * np.array(flux[cons_var])

    return var, flux_rkt

def update_field(var, flux):
    rho_temp  = - flux[0] * dt + var.rhoc
    rhou_temp = - flux[1] * dt + var.rhoc * var.uc
    rhov_temp = - flux[2] * dt + var.rhoc * var.vc
    e_temp    = - flux[3] * dt + var.rhoc * 0.5 * (var.uc**2 + var.vc**2) + var.pc / (shr - 1)

    var.rhoc = rho_temp
    var.uc   = rhou_temp / rho_temp
    var.vc   = rhov_temp / rho_temp
    var.pc   = (shr-1) * (e_temp - 0.5 * rho_temp * (var.uc**2 + var.vc**2))
    var.Tc   = shr * var.pc / rho_temp

    return var

def cal_outer_res(*var_pairs):
    errors = []
    vars_old = []
    for var, var_old in var_pairs:
        denom = np.sum(np.abs(var))
        if denom > 1e-9:
            error = np.sum(np.abs(var - var_old)) / denom
        else:
            error = 0.0
        errors.append(error)
        vars_old.append(var.copy())
    return errors, vars_old

