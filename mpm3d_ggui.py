import numpy as np

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 64, 5, 4e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 5, 1e-4

n_particles = n_grid**dim // 2**(dim - 1)

print(n_particles)

dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
g_x = 0
g_y = -9.8
g_z = 0
bound = 3
# @ti.func
# def bound (xg_x: float) -> int:
#     x = ti.cast(xg_x, int)
#     terrian = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16}
#     return terrian[x]

E = 1000  # Young's modulus
nu = 0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(3, 3, dtype=float,
                    shape=n_particles)  # deformation gradient
Jp = ti.field(float, n_particles)

colors = ti.Vector.field(4, float, n_particles)
colors_random = ti.Vector.field(4, float, n_particles)
materials = ti.field(int, n_particles)
grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)
used = ti.field(int, n_particles)



neighbour = (3, ) * dim
neighbour1 = (4, ) * dim


WATER = 0
JELLY = 1
SNOW = 2
SOIL = 3

@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(grid_m): #sets all grid masses and velocities to 0
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    #p is index, not the particle itself
    for p in x:
        if materials[p] == SNOW:
            #MPM step 1, setting grid masses and velocities
            Xp = x[p] / dx #scaling particles position to match grid space?
            baseX = int(Xp[0]) 
            baseY = int(Xp[1])
            baseZ = int(Xp[2])
            diffX = Xp[0] - baseX
            diffY = Xp[1] - baseY
            diffZ = Xp[2] - baseZ
            #checks grid points up to 2 grid points away since N will be 0 where the distance between the points position and grid points is over 2
            for offset in ti.static(ti.grouped(ti.ndrange(*neighbour1))):
                #don't try to access negative grid indices
                if (baseX + offset[0] - 1 < 0):
                    continue
                if (baseY + offset[1] - 1 < 0):
                    continue 
                if (baseZ + offset[2] - 1 < 0):
                    continue
                #distance between particle position and grid position
                diffX = baseX - (baseX + offset[0] - 1)
                diffY = baseY - (baseY + offset[1] - 1)
                diffZ = baseZ - (baseZ + offset[2] - 1)
                abs_diffX = abs(diffX)
                abs_diffY = abs(diffY)
                abs_diffZ = abs(diffZ)

                distances = ti.Vector([abs_diffX, abs_diffY, abs_diffZ])

                weight = 1.0
                #equation to be used if distance between particle is < 1, w01[0] is the calcuation for the x position, w01[1] is calculation for the y position, etc
                w01 = [0.5 * (abs_diffX)**3 - diffX**2 + 2.0 / 3.0,0.5 * (abs_diffY)**3 - diffY**2 + 2.0 / 3.0, 0.5 * (abs_diffZ)**3 - diffZ**2 + 2.0 / 3.0]
                #equation to be used if distance between particle is > 1, w12[0] is the calcuation for the x position, w12[1] is calculation for the y position, etc (implicit less than 2 from offsets used)
                w12 = [-1.0/6.0 * (abs_diffX)**3 + diffX**2 - 2 * abs_diffX + 4.0 / 3.0, -1.0/6.0 * (abs_diffY)**3 + diffY**2 - 2 * abs_diffY + 4.0 / 3.0, -1.0/6.0 * (abs_diffZ)**3 + diffZ**2 - 2 * abs_diffZ + 4.0 / 3.0]
                for i in ti.static(range(dim)):
                    if distances[i] < 1:
                        weight *= w01[i]
                    else:
                        weight *= w12[i]
                grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * p_mass
                grid_v[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1] += weight * (p_mass * v[p]) / grid_m[baseX + offset[0] - 1, baseY + offset[1] - 1, baseZ + offset[2] - 1]
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

    #  particle interactions
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        
        cond = (I < bound) & (grid_v[I] < 0) | \
               (I > n_grid - bound) & (grid_v[I] > 0)
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
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

terrain = []
for xx in np.linspace(0.05, 0.9, num = 10):
    for yy in np.linspace(0.05, 0.9, num = 10):
        z = np.random.choice(np.linspace(0, 0.5, num = 10), size = 1)[0]
        single_terrian = CubeVolume(ti.Vector([xx, 0, yy]),
                          ti.Vector([0.1, z, 0.1]), SOIL)
        terrain.append(single_terrian)

snowfall = [
               CubeVolume(ti.Vector([0.05, 0.8, 0.05]),
                          ti.Vector([0.95, 0.2, 0.95]), SNOW),
            ]

snowfall.extend(terrain)

presets = [ snowfall,
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
    "Snow Falling Mountain",
    "Double Dam Break",
    "Water Snow Jelly",
    "snow",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.01

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0), (0.2, 0.6, 0.2)]


def init():
    global paused
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

    window.GUI.begin("Presets", 0.05, 0.1, 0.2, 0.15)
    old_preset = curr_preset_id
    for i in range(len(presets)):
        if window.GUI.checkbox(preset_names[i], curr_preset_id == i):
            curr_preset_id = i
    if curr_preset_id != old_preset:
        init()
        paused = True
    window.GUI.end()

    window.GUI.begin("Gravity", 0.05, 0.3, 0.2, 0.1)
    g_x = window.GUI.slider_float("x", g_x, -10, 10)
    g_y = window.GUI.slider_float("y", g_y, -10, 10)
    g_z = window.GUI.slider_float("z", g_z, -10, 10)
    window.GUI.end()

    window.GUI.begin("Options", 0.05, 0.45, 0.2, 0.4)

    use_random_colors = window.GUI.checkbox("use_random_colors",
                                            use_random_colors)
    if not use_random_colors:
        material_colors[WATER] = window.GUI.color_edit_3(
            "water color", material_colors[WATER])
        material_colors[SNOW] = window.GUI.color_edit_3(
            "snow color", material_colors[SNOW])
        material_colors[JELLY] = window.GUI.color_edit_3(
            "jelly color", material_colors[JELLY])
        set_color_by_material(np.array(material_colors, dtype=np.float32))
    particles_radius = window.GUI.slider_float("particles radius ",
                                               particles_radius, 0, 0.1)
    if window.GUI.button("restart"):
        init()
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = colors_random if use_random_colors else colors
    scene.particles(x, per_vertex_color=colors_used, radius=particles_radius)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)


while window.running:
    #print("heyyy ",frame_id)
    frame_id += 1
    frame_id = frame_id % 256

    if not paused:
        for s in range(steps):
            # bound = np.random.randint(2, high = 20)
            substep(g_x, g_y, g_z)

    render()

    show_options()

    window.show()