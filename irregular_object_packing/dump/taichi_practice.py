"""

This file includes a cloth and ball simulation that has been written based on the following medium article below.
It is written as a practice for the Taichi programming language.
medium article: https://medium.com/parallel-programming-in-python/head-first-taichi-a-beginners-guide-to-high-performance-computing-in-python-be6afc5db93e

"""
# %%
import taichi as ti

ti.init(arch=ti.gpu)
# %%
N = 128
x = ti.Vector.field(3, float, (N, N))
v = ti.Vector.field(3, float, (N, N))

ball_center = ti.Vector.field(3, float, (1,))

stiffness = 1600
damping = 2
ball_radius = 0.2
gravity = 0.5
dt = 5e-4
cell_size = 1.0 / N


links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = [ti.Vector(v) for v in links]


# %% Initialisation of values in ti.Vectors
def init_scene():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector(
            [
                i * cell_size,
                j * cell_size / ti.sqrt(2),
                (N - j) * cell_size / ti.sqrt(2),
            ]
        )
    ball_center[0] = ti.Vector([0.5, -0.5, -0.0])


# %% Computation of the simulation


@ti.kernel
def step_no_slide():
    """This is the main iteration of the simulation. It is basically a verlet integration,
    where the position of the particles is updated based on the velocity and the acceleration.
    The acceleration is calculated based on the forces that are applied to the particles.
    """
    # first, update the vertical velocity based on the gravity constant
    for i in ti.grouped(x):
        v[i].y -= gravity * dt

    # then, we add the velocity due to the force of the springs defined between the particles
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        # loop over all the neighbours of the particle i
        for d in ti.static(links):
            j = ti.min(ti.max(i + d, 0), [N - 1, N - 1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i - j).norm()
            if original_length != 0:
                force += (
                    stiffness
                    * relative_pos.normalized()
                    * (current_length - original_length)
                    / original_length
                )
        v[i] += force * dt

    # update the speed of the particles based on the damping constant
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)

        # set speed to zero if the particle is at the surface of the ball
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            v[i] = ti.Vector([0.0, 0.0, 0.0])

        # update the position of the particles based on the final speed for this iteration
        x[i] += dt * v[i]


@ti.kernel
def step_slide():
    """This is the main iteration of the simulation. It is basically a verlet integration,
    where the position of the particles is updated based on the velocity and the acceleration.
    The acceleration is calculated based on the forces that are applied to the particles.
    """
    # first, update the vertical velocity based on the gravity constant
    for i in ti.grouped(x):
        v[i].y -= gravity * dt

    # then, we add the velocity due to the force of the springs defined between the particles
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        # loop over all the neighbours of the particle i
        for d in ti.static(links):
            j = ti.min(ti.max(i + d, 0), [N - 1, N - 1])
            relative_pos = x[j] - x[i]
            current_length = relative_pos.norm()
            original_length = cell_size * float(i - j).norm()
            if original_length != 0:
                force += (
                    stiffness
                    * relative_pos.normalized()
                    * (current_length - original_length)
                    / original_length
                )
        v[i] += force * dt

    # update the speed of the particles based on the damping constant
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)

        # set speed to zero if the particle is at the surface of the ball
        if (x[i] - ball_center[0]).norm() <= ball_radius:
            # v[i] = ti.Vector([0.0, 0.0, 0.0])
            # # for sliding, we want the velocity in de direction of the ball radius to be zero,
            # so we project the velocity vector on the normal vector of the ball surface,
            # this is the velocity in the direction of the ball radius. that we need to substract
            n_c = x[i] - ball_center[0]  # normal vector on ball surface
            proj_n_v = v[i].dot(n_c) * n_c / ball_radius
            v[i] = v[i] - proj_n_v

            x[i] = x[i] + (x[i] - ball_center[0])

            # v[i] = v[i] - (v[i].dot(x[i] - ball_center[0])) * (x[i] - ball_center[0]) / ball_radius**2

        # update the position of the particles based on the final speed for this iteration
        x[i] += dt * v[i]


# %% Rendering of the simulation
# The cloth will be rpresented as a triangle mesh, which for a square grid is just a grid
# of triangles, where each cell ■ exists of two triangles. ◣ & ◥.
num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)


@ti.kernel
def set_vertices():
    """Sets the vertices of the mesh to the positions of the particles.
    The `i * N` is because the mesh is a 1D array of vertices,
    and we need to convert the 2D indices to 1D."""
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]


@ti.kernel
def set_indices():
    """Sets the indices of the mesh grid points to the vertices of the triangles."""
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j


# %%
init_scene()
set_indices()

window = ti.ui.Window("Cloth Simulation", (800, 800), vsync=False)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
while window.running:
    for _i in range(30):
        step_no_slide()
    set_vertices()
    camera.position(0.5, -0.5, 2)
    camera.lookat(0.5, -0.5, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
    scene.mesh(vertices, indices=indices, color=(0.5, 0.5, 0.5), two_sided=True)
    scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0.0))
    canvas.scene(scene)
    window.show()
