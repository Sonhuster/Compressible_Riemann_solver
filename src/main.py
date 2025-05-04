import copy
import numpy as np

import config as cf
import utils as ufs
import solver as sol
import initialization as init
from input import *


if __name__ == '__main__':
    if READ_MESH is False and SIMULATION_MODE in (0, 2, 3):
        coordinates = ufs.get_coordinate(Lx=2.0, Ly=1.0, nx=20, ny=20)
        mesh = cf.from_coords_to_mesh(coordinates, noise, element_type = "TRI")     # Element type "UPPER CASE"
        del coordinates
    elif READ_MESH is True and SIMULATION_MODE in (1, ):
        mesh = cf.loadfile(refined_level=0)
    else:
        raise "Incorrectly defined input file"

    # Node and face weighted coefficients
    fw = cf.cell_to_face_interpolation(mesh)
    cw = cf.cell_to_node_interpolation(mesh)

    # Allocate memory && initialization
    print("_______________________START INITIALIZATION_______________________")
    var = init.Fluid(mesh, 0.0, v_inf, p_inf, T_inf, rho)
    var_rkt = copy.deepcopy(var)
    state_L, state_R = init.init_LR_state(mesh)
    print("_______________________FINISH INITIALIZATION_______________________")

    uc_old, vc_old, pc_old = var.uc.copy(), var.vc.copy(), var.pc.copy()
    conv_flux = [np.zeros(mesh.no_faces()) for cons_var in range(4)]
    diff_flux = [np.zeros(mesh.no_faces()) for cons_var in range(4)]

    print("_______________________START SOLVING_______________________")
    for iter_ in range(iter_outer):
        flux_rkt = [np.zeros(mesh.no_elems()) for cons_var in range(4)]   # Should be (5, N) if 3D.
        for rk_idx, rk in enumerate(rk4):
            var_rkt = sol.cal_node_face_value(mesh, var_rkt, fw, cw) if rk_idx > 0 else sol.cal_node_face_value(mesh, var, fw, cw)

            # Left-Right state (1st order)
            state_L, state_R = sol.cal_left_right_state(mesh, fw, var_rkt)
            # Convective flux
            if convection_scheme == 0:
                conv_flux = sol.ROE_flux(mesh, state_L, state_R)
            elif convection_scheme == 1:
                conv_flux = sol.AUSM_flux(mesh, state_L, state_R)
            else:
                "Convection scheme should be ROE||AUSM"
            # Diffusion flux (2nd gradient)
            diff_flux = sol.diffusion_flux(mesh, var_rkt, state_L, state_R)

            # Flux summation
            flux = sol.intergrate_flux(mesh, conv_flux, diff_flux)

            # Calculate primitive variables
            var_rkt, flux_rkt = sol.cal_rk_temporal_var(var_rkt, flux, flux_rkt, rk)

        var = sol.update_field(var, flux_rkt)

        # Calculate residual
        [(error_u, error_v, error_p), (uc_old, vc_old, pc_old)] = sol.cal_outer_res((var.uc, uc_old), (var.vc, vc_old), (var.pc, pc_old))
        print(f"Outer iteration {iter_} - residual (u, v, p) = ({error_u}, {error_v}, {error_p})")
        if error_u < 1e-05 and error_v < 1e-05:
            break

        if iter_ % int(n_plot) == 0:
            filename = f"../paraview/para_{iter_}"
            var = sol.cal_node_face_value(mesh, var, fw, cw)
            ufs.write_vtk_with_streamline([var.uv, var.vv, var.pv, var.Tv], mesh,
                                          ["Velocity U", "Velocity V", "Pressure", "Temperature"], filename)

    # var = sol.cal_node_face_value(mesh, var, fw, cw)
    # ufs.plot_vtk(var.uv, mesh, "Velocity U")
    # ufs.plot_vtk(var.vv, mesh, "Velocity V")
    # ufs.plot_vtk(var.pv, mesh, "Pressure")
    # ufs.plot_vtk(var.Tv, mesh, "Temperature")
    print("_______________________FINISH CASE_______________________")
