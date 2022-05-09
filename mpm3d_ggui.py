import numpy as np
import random
import math
import taichi as ti
from scipy.stats import multivariate_normal

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt, init_v_x, init_v_y, init_v_z = 3, 64, 20, 4e-4, 0, 0, 0
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 400
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
g_x = 0
g_y = -9.8
g_z = 0
bound = 3
wind_strength = 10

# @ti.func
# def bound (xg_x: float) -> int:
#     x = ti.cast(xg_x, int)
#     terrian = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16}
#     return terrian[x]

alpha = 0.95
E = 140000  # Young's modulus
nu = 0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
xi = 10 #snow's hardening coeficient
theta_c = 0.025
theta_s = 0.0075
x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(3, 3, dtype=float,
                    shape=n_particles)  # deformation gradient
F_E = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) #deformation gradient for E
F_P = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
F_hat_ep = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
Jp = ti.field(float, n_particles)

colors = ti.Vector.field(4, float, n_particles)
colors_random = ti.Vector.field(4, float, n_particles)
materials = ti.field(int, n_particles)
grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_v_old = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)
grid_f = ti.Vector.field(dim, float, (n_grid, ) * dim)
used = ti.field(int, n_particles)



neighbour = (3, ) * dim
grid_offsets = (4, ) * dim


WATER = 0
JELLY = 1
SNOW = 2
SOIL = 3

@ti.func
def lame_mu(mu_0, xi, J_p):
    return mu_0 * ti.exp(xi*(1-J_p))

@ti.func
def lame_lambda(lambda_0, xi, J_p):
    return lambda_0 * ti.exp(xi*(1-J_p))


@ti.func
def psi_derivative(mu_0, lambda_0, xi, p):
    J_p = F_P[p].determinant()
    J_e = F_hat_ep[p].determinant()
    RE, _ = ti.polar_decompose(F_hat_ep[p])
    return (2.0 * lame_mu(mu_0, xi, J_p) * (F_hat_ep[p]-RE) + lame_lambda(lambda_0, xi, J_p) * (J_e-1) * J_e * F_hat_ep[p].transpose().inverse()) / J_p
@ti.func
def clamp(sigma, theta_c, theta_s):
    ret = ti.Matrix([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])
    for d in ti.static(range(3)):
        ret[d, d] = min(max(sigma[d, d], 1 - theta_c), 1 + theta_s)
    return ret


@ti.func
def compute_weight(Xp, offset, baseX, baseY, baseZ):

    diffX = Xp[0] - (baseX + offset[0] - 1)
    diffY = Xp[1] - (baseY + offset[1] - 1)
    diffZ = Xp[2] - (baseZ + offset[2] - 1)
    diff = ti.Vector([diffX, diffY, diffZ])

    abs_diffX = abs(diffX)
    abs_diffY = abs(diffY)
    abs_diffZ = abs(diffZ)

    distances = ti.Vector([abs_diffX, abs_diffY, abs_diffZ])

    #equation to be used if distance between particle is < 1, w01[0] is the calcuation for the x position, w01[1] is calculation for the y position, etc
    w01 = ti.Vector([0.5 * (abs_diffX)**3 - diffX**2 + 2.0 / 3.0,0.5 * (abs_diffY)**3 - diffY**2 + 2.0 / 3.0, 0.5 * (abs_diffZ)**3 - diffZ**2 + 2.0 / 3.0])
    #equation to be used if distance between particle is > 1, w12[0] is the calcuation for the x position, w12[1] is calculation for the y position, etc (implicit less than 2 from offsets used)
    w12 = ti.Vector([-1.0/6.0 * (abs_diffX)**3 + diffX**2 - 2 * abs_diffX + 4.0 / 3.0, -1.0/6.0 * (abs_diffY)**3 + diffY**2 - 2 * abs_diffY + 4.0 / 3.0, -1.0/6.0 * (abs_diffZ)**3 + diffZ**2 - 2 * abs_diffZ + 4.0 / 3.0])

    weight = 1.0
    for i in ti.static(range(dim)):
        if distances[i] < 1:
            weight *= w01[i]
        else:
            weight *= w12[i]
    return weight, w01, w12, diff

@ti.func
def compute_weight_grad(w01, w12, diff):

    diffX = diff[0]
    diffY = diff[1]
    diffZ = diff[2]
    abs_diffX = abs(diffX)
    abs_diffY = abs(diffY)
    abs_diffZ = abs(diffZ)

    
    diffX_sign = 1 if (diffX > 0) else -1
    diffY_sign = 1 if (diffY > 0) else -1
    diffZ_sign = 1 if (diffZ > 0) else -1


    distances = ti.Vector([abs_diffX, abs_diffY, abs_diffZ])
    #derivative of w01
    w01d = [1.5 * diffX ** 2 * diffX_sign - 2 * diffX, 1.5 * diffY **2 * diffY_sign - 2 * diffY, 1.5 * diffZ **2 * diffZ_sign - 2 * diffZ]
    #derivative of w12
    w12d = [-0.5 * diffX ** 2 * diffX_sign + 2 * diffX - 2 * diffX_sign, -0.5 * diffY ** 2 * diffY_sign + 2 * diffY - 2 * diffY_sign, -0.5 * diffZ ** 2 * diffZ_sign + 2 * diffZ - 2 * diffZ_sign]

    #gradient of w in the case distance is >=0 <1
    x_w = 100.0
    x_wdx = 100.0

    y_w = 100.0
    y_wdy = 100.0

    z_w = 100.0
    z_wdz = 100.0
    if distances[0] < 1:
        x_w = w01[0]
        x_wdx = w01d[0]
    else:
        x_w = w12[0]
        x_wdx = w12d[0]

    if distances[1] < 1:
        y_w = w01[1]
        y_wdy = w01d[1]
    else:
        y_w = w12[1]
        y_wdy = w12d[1]

    if distances[2] < 1:
        z_w = w01[2]
        z_wdz = w01d[2]
    else:
        z_w = w12[2]
        z_wdz = w12d[2]




    w_grad = ti.Vector([y_w * z_w * x_wdx, x_w * z_w * y_wdy, x_w * y_w * z_wdz])

    return w_grad / dx

@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(grid_m): #sets all grid masses and velocities to 0, v_old keeps old grid velocities
        grid_v_old[I] = grid_v[I]
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
        grid_f[I] = ti.zero(grid_f[I])
    ti.block_dim(n_grid)
    #p is index, not the particle itself
    for p in x:
        if materials[p] == SNOW:
            #MPM step 1, setting grid masses and velocities
            Xp = x[p] / dx #scaling particles position to match grid space?
            baseX = int(Xp[0])
            baseY = int(Xp[1])
            baseZ = int(Xp[2])
            #diffX = Xp[0] - baseX
            #diffY = Xp[1] - baseY
            #diffZ = Xp[2] - baseZ
            summation = ti.Matrix([[0,0,0], [0,0,0], [0,0,0]], ti.f32)
            #checks grid points up to 2 grid points away since N will be 0 where the distance between the points position and grid points is over 2
            for offset in ti.grouped(ti.ndrange(*grid_offsets)):
                #don't try to access negative grid indices
                if (baseX + offset[0] - 1 < 0):
                   continue
                if (baseY + offset[1] - 1 < 0):
                   continue
                if (baseZ + offset[2] - 1 < 0):
                   continue
                #distance between particle position and grid position
                weight, _, _, _ = compute_weight(Xp, offset, baseX, baseY, baseZ)

                grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * p_mass
                grid_v[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * p_mass * v[p] #/ grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
    for p in x:
        if materials[p] == SNOW:
            #MPM step 3
            Xp = x[p] / dx #scaling particles position to match grid space?
            baseX = int(Xp[0])
            baseY = int(Xp[1])
            baseZ = int(Xp[2])
            #diffX = Xp[0] - baseX
            #diffY = Xp[1] - baseY
            #diffZ = Xp[2] - baseZ
            summation = ti.Matrix([[0,0,0], [0,0,0], [0,0,0]], ti.f32)
            for offset in ti.grouped(ti.ndrange(*grid_offsets)):
                #don't try to access negative grid indices
                if (baseX + offset[0] - 1 < 0):
                   continue
                if (baseY + offset[1] - 1 < 0):
                   continue
                if (baseZ + offset[2] - 1 < 0):
                   continue
                weight, w01, w12, diff = compute_weight(Xp, offset, baseX, baseY, baseZ)

                
                #gradient of w in the case distance is >=0 <1
                w_gradient = compute_weight_grad(w01, w12, diff)

                velocity = grid_v[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
                #we hadn't yet scaled grid velocity by grid mass in step 1 because not all grid masses were updated yet
                if (grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] > 0):
                    velocity /= grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
                summation += velocity.outer_product(w_gradient)
            identity = ti.Matrix([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], ti.f32)
            F_hat_ep[p] = (identity + dt * summation) @ F_E[p]

            sigma_p = psi_derivative(mu_0, lambda_0, xi, p) @ F_E[p].transpose()
            neg_force_unweighted = p_vol * sigma_p
            #print('neg_force_unweighted', neg_force_unweighted)
            for offset in ti.grouped(ti.ndrange(*grid_offsets)):
                #don't try to access negative grid indices
                if (baseX + offset[0] - 1 < 0):
                   continue
                if (baseY + offset[1] - 1 < 0):
                   continue
                if (baseZ + offset[2] - 1 < 0):
                   continue
                weight, w01, w12, diff = compute_weight(Xp, offset, baseX, baseY, baseZ)

                
                #gradient of w in the case distance is >=0 <1
                w_gradient = compute_weight_grad(w01, w12, diff)
                grid_f[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] -= neg_force_unweighted @ w_gradient
                #for i in ti.static(range(dim)):
                #    if distances[i] < 1:
                #        weight *= w01[i]
                #    else:
                #        weight *= w12[i]
                #grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * p_mass
                #grid_v[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * (p_mass * v[p]) / grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
            #MPM step 4
        else:
            if used[p] == 0:
                continue
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

            F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # deformation gradient update

            h = ti.exp(10 * (1.0 - Jp[p]))  # Hardening coefficient: snow gets harder when compressed
            if materials[p] == JELLY:  # jelly, make it softer
                h = 0.3
            mu, la = mu_0 * h, lambda_0 * h
            if materials[p] == WATER:  # liquid
                mu = 0.0

            U, sig, V = ti.svd(F[p])
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                if materials[p] == SNOW:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if materials[p] == WATER:  # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(float, 3)
                new_F[0, 0] = J
                F[p] = new_F
            elif materials[p] == SNOW:
                F[p] = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
            stress = (-dt * p_vol * 4) * stress / dx**2
            affine = stress + p_mass * C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    #MPM step 4 and 6
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
            grid_v[I] += dt / grid_m[I] * grid_f[I]
        grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        
        cond = (I < bound) & (grid_v[I] < 0) | \
               (I > n_grid - bound) & (grid_v[I] > 0)
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        if materials[p] == SNOW:
            Xp = x[p] / dx #scaling particles position to match grid space?
            baseX = int(Xp[0])
            baseY = int(Xp[1])
            baseZ = int(Xp[2])
            new_v_pic = ti.zero(v[p])
            new_v_flip = ti.zero(v[p])
            #MPM step 7 and 8
            grad_v = ti.Matrix([[0,0,0], [0,0,0], [0,0,0]], ti.f32)
            for offset in ti.grouped(ti.ndrange(*grid_offsets)):
                if (baseX + offset[0] - 1 < 0):
                   continue
                if (baseY + offset[1] - 1 < 0):
                   continue
                if (baseZ + offset[2] - 1 < 0):
                   continue
                weight, w01, w12, diff = compute_weight(Xp, offset, baseX, baseY, baseZ)

                
                #gradient of w in the case distance is >=0 <1
                w_gradient = compute_weight_grad(w01, w12, diff)

                g_v_new = grid_v[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
                grad_v += g_v_new.outer_product(w_gradient)
                #mpm step 8
                g_v_old = grid_v_old[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
                new_v_pic += weight * g_v_new
                new_v_flip += weight * (g_v_new - g_v_old)
            #mpm step 7

            F_hat_ep_next = (ti.Matrix([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], ti.f32) + dt * grad_v) @ F_E[p]
            F[p] = F_hat_ep_next @ F_P[p]
            U, Sigma, V = ti.svd(F_hat_ep_next)
            Sigma = clamp(Sigma, theta_c, theta_s)
            F_E[p] = U @ Sigma @ V.transpose()
            F_P[p] = V @ Sigma.inverse() @ U.transpose() @ F[p]

            #mpm step 10
            v[p] = (1.0 - alpha) * new_v_pic + alpha * (v[p] + new_v_flip)
            x[p] += dt * v[p]
        else:
            if materials[p] == SOIL:
                continue
            if used[p] == 0:
                continue
            Xp = x[p] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.zero(v[p])
            new_C = ti.zero(C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
                dpos = (offset - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
            v[p] = new_v
            x[p] += dt * v[p]
            C[p] = new_C


class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        #x[i] is set to a random position in the bounds of the volume of the cube representing the position of particle i
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        Jp[i] = 1
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_E[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_P[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_hat_ep[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #v[i] represents the velocity of particle i
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        materials[i] = material
        colors_random[i] = ti.Vector(
            [ti.random(), ti.random(),
             ti.random(), ti.random()])
        used[i] = 1


@ti.kernel
def set_all_unused():
    for p in used:
        used[p] = 0
        # basically throw them away so they aren't rendered
        x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        Jp[p] = 1
        F[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        v[p] = ti.Vector([0.0, 0.0, 0.0])
    for I in ti.grouped(grid_m): #sets all grid masses and velocities to 0, v_old keeps old grid velocities
        grid_v[I] = ti.zero(grid_v[I])



def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume
    next_p = 0
    for i in range(len(vols)):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(
                    vols
            ) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size,
                          v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(material_colors: ti.types.ndarray()):
    for i in range(n_particles):
        mat = materials[i]
        colors[i] = ti.Vector([
            material_colors[mat, 0], material_colors[mat, 1],
            material_colors[mat, 2], 1.0
        ])


print("Loading presets...this might take a minute")

MAX_SLOPE = 45
MIN_SLOPE = -45
MIN_HEIGHT = 0.1

def dist_squared(P1,P2):
    return (P1[0]-P2[0])**2 + (P1[1]-P2[1])**2

def mountain(P1,P2, result):
    if dist_squared(P1,P2) < 0.03:
        result.append(P2)
        return
    x1,y1 = P1
    x2,y2 = P2
    x3 = random.uniform(x1,x2)
    y3_max = min((x3-x1)*math.tan(math.radians(MAX_SLOPE)) + y1, (x2-x3)*math.tan(-math.radians(MIN_SLOPE)) + y2)
    y3_min = max((x3-x1)*math.tan(math.radians(MIN_SLOPE)) + y1, (x2-x3)*math.tan(-math.radians(MAX_SLOPE)) + y2)
    y3_min = max(y3_min, MIN_HEIGHT)
    y3 = random.uniform(y3_min,y3_max)
    P3 = (x3, y3)
    mountain(P1,P3, result)
    mountain(P3,P2, result)
    return

heights = []
mountain((0, 0.1), (1, 0.4), heights)
# print(heights)

# Gausian 

Gausian_x, Gausian_y = np.mgrid[0.0:0.95:20j, 0.0:0.95:20j]
Gausian_xx = Gausian_x[:,0]
Gausian_yy = Gausian_y[0,:]
Gausian_xy = np.column_stack([Gausian_x.flat, Gausian_y.flat])
mu1 = np.array([0.25, 0.25])
sigma1 = np.array([.05, .05])
covariance1 = np.diag(sigma1**1)
mu2 = np.array([0.75, 0.75])
sigma2 = np.array([.05, .05])
covariance2 = np.diag(sigma2**1)
z1 = multivariate_normal.pdf(Gausian_xy, mean=mu1, cov=covariance1) / 6
z2 = multivariate_normal.pdf(Gausian_xy, mean=mu2, cov=covariance2) / 8
Gausian_z = z1 + z2
Gausian_z = Gausian_z.reshape(Gausian_x.shape)

Gausian_terrain = []
# Gaussian Mountain Generation
for i in range(len(Gausian_xx)):
    for j in range(len(Gausian_yy)):
        zz = Gausian_z[i][j] + 0.1
        single_terrian = CubeVolume(ti.Vector([Gausian_xx[i], 0, Gausian_yy[j]]),
                            ti.Vector([0.05, zz, 0.05]), SOIL)
        Gausian_terrain.append(single_terrian)

Gausian_x, Gausian_y = np.mgrid[0.0:0.975:40j, 0.0:0.975:40j]
Gausian_xx = Gausian_x[:,0]
Gausian_yy = Gausian_y[0,:]
Gausian_xy = np.column_stack([Gausian_x.flat, Gausian_y.flat])
Gausian_mu = np.array([0.5, 0.5])
Gausian_sigma = np.array([.1, .1])
Gausian_covariance = np.diag(Gausian_sigma**1)
Gausian_z = multivariate_normal.pdf(Gausian_xy, mean=Gausian_mu, cov=Gausian_covariance) / 2
Gausian_z = Gausian_z.reshape(Gausian_x.shape)
# Gaussian Mountain Generation
Gausian_terrain_single = []
for i in range(len(Gausian_xx)):
    for j in range(len(Gausian_yy)):
        zz = Gausian_z[i][j] - 0.5
        if zz > 0.2:
            single_terrian = CubeVolume(ti.Vector([Gausian_xx[i], 0.1, Gausian_yy[j]]),
                                ti.Vector([0.05, zz, 0.05]), SNOW)
            Gausian_terrain_single.append(single_terrian)
        # else:
        #     zz = 0.1
        #     single_terrian = CubeVolume(ti.Vector([Gausian_xx[i], 0, Gausian_yy[j]]),
        #                         ti.Vector([0.05, zz, 0.05]), WATER)
        #     Gausian_terrain_single.append(single_terrian)

Gausian_terrain_single.append(CubeVolume(ti.Vector([0.6, 0.05, 0.6]),
                          ti.Vector([0.25, 0.25, 0.25]), WATER),)

terrain = []

for i in range(len(heights) - 1):
    # for yy in np.linspace(0, 0.9, num = 10):
        # z = np.random.choice(np.linspace(0, 0.5, num = 10), size = 1)[0]
        xx = heights[i][0]
        zz = heights[i][1]
        single_terrian = CubeVolume(ti.Vector([xx, 0, 0]),
                          ti.Vector([heights[i + 1][0] - xx, zz, 1]), SOIL)
        terrain.append(single_terrian)

snowfall1 = [
               CubeVolume(ti.Vector([0.05, 0.95, 0.05]),
                          ti.Vector([0.95, 0.1, 0.95]), SNOW),
            ]

snowfall2 = [
               CubeVolume(ti.Vector([0.2, 0.6, 0.2]),
                          ti.Vector([0.6, 0.1, 0.6]), SNOW),
            ]

snowfall_init_speed = [
               CubeVolume(ti.Vector([0.2, 0.6, 0.2]),
                          ti.Vector([0.6, 0.1, 0.6]), SNOW, ),
            ]

snow_melting = [
               CubeVolume(ti.Vector([0.2, 0.6, 0.2]),
                          ti.Vector([0.6, 0.1, 0.6]), SNOW, ),
            ]

snowfall1.extend(Gausian_terrain)
snowfall2.extend(terrain)
snowfall_init_speed.extend(Gausian_terrain)

presets = [ snowfall1,
            snowfall2,
            snowfall_init_speed,
            Gausian_terrain_single,
           [
               CubeVolume(ti.Vector([0.05, 0.05, 0.05]),
                          ti.Vector([0.3, 0.4, 0.3]), WATER),
               CubeVolume(ti.Vector([0.65, 0.05, 0.65]),
                          ti.Vector([0.3, 0.4, 0.3]), SOIL),
           ],
           [
               CubeVolume(ti.Vector([0.6, 0.05, 0.6]),
                          ti.Vector([0.25, 0.25, 0.25]), WATER),
               CubeVolume(ti.Vector([0.35, 0.35, 0.35]),
                          ti.Vector([0.25, 0.25, 0.25]), SNOW),
               CubeVolume(ti.Vector([0.05, 0.6, 0.05]),
                          ti.Vector([0.25, 0.25, 0.25]), JELLY),
           ],
           [
    CubeVolume(ti.Vector([0.1, 0.1, 0.1]), ti.Vector([0.8, 0.1, 0.8]),
               SNOW),
    CubeVolume(ti.Vector([0.25, 0.7, 0.25]),
                          ti.Vector([0.24, 0.24, 0.24]), JELLY),
]]
preset_names = [
    "Snow Falling",
    "Snow Sheet Falling - Step Mountain",
    "Snow Sheet Falling - Gaussian Mountain",
    "Snow Melting In Water",
    "Tsunami",
    "Water Snow Jelly",
    "snow",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.01

use_random_wind = False

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0), (155/255, 118/255, 83/255)]


def init():
    global paused
    if curr_preset_id == 3:
        init_v_x, init_v_y, init_v_z = 5, 0, 0
        init_vols(presets[curr_preset_id])
    else:
        init_vols(presets[curr_preset_id])


init()

res = (1920, 1080)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id
    global g_x, g_y, g_z
    global wind_strength
    global use_random_wind

    window.GUI.begin("Presets", 0.05, 0.1, 0.2, 0.25)
    old_preset = curr_preset_id
    for i in range(len(presets)):
        if window.GUI.checkbox(preset_names[i], curr_preset_id == i):
            curr_preset_id = i
    if curr_preset_id != old_preset:
        init()
        paused = True
    window.GUI.end()

    window.GUI.begin("Wind and Gravity", 0.05, 0.3, 0.2, 0.2)
    wind_strength = window.GUI.slider_float("Wind Strength", wind_strength, 0, 20)
    g_y = window.GUI.slider_float("Gravity Constant", g_y, -10, 10)
    particles_radius = window.GUI.slider_float("Particles Radius ",
                                               particles_radius, 0, 0.1)
    use_random_wind = window.GUI.checkbox("Use Random Wind",
                                            use_random_wind)
    if window.GUI.button("restart"):
        init()
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    set_color_by_material(np.array(material_colors, dtype=np.float32))
    
    window.GUI.end()


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = colors_random if use_random_colors else colors
    scene.particles(x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 0.8, -0.5), color=(0.8, 0.5, 0.0))
    scene.point_light(pos=(-0.5, 0.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(1.5, 0.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 0.8, 1.5), color=(0.5, 0.5, 0.5))
    # scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.99))

    canvas.scene(scene)


while window.running:
    frame_id += 1
    frame_id = frame_id % 256
    # wind simulation
    prob = [True] + [False] * 10
    # if np.random.choice(prob):
    #     g_x = random.randint(-round(wind_strength), round(wind_strength))
    if not paused:
        for s in range(steps):
            # bound = np.random.randint(2, high = 20)
            substep(g_x, g_y, g_z)
    
    
    if use_random_wind:
        if np.random.choice(prob):
            g_x += random.randint(-round(wind_strength), round(wind_strength))
            g_x = max(g_x, -round(wind_strength))
            g_x = min(g_x, round(wind_strength))
    else:
        mouse_pos = window.get_cursor_pos()
        g_x = (0.5 - mouse_pos[0]) * wind_strength
    render()
    show_options()
    window.show()
