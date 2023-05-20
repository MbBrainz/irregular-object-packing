# %%
import logging
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from concurrent.futures import wait
from time import sleep, time

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import PolyData
from tqdm.auto import tqdm

from irregular_object_packing.cat import chordal_axis_transform as cat
from irregular_object_packing.cat.cat_data import CatData
from irregular_object_packing.cat.tetra_cell import filter_relevant_cells
from irregular_object_packing.cat.utils import get_cell_arrays
from irregular_object_packing.mesh.collision import (
    compute_all_collisions,
    compute_container_violations,
    compute_object_collisions,
)
from irregular_object_packing.mesh.sampling import (
    mesh_simplification_condition,
    resample_mesh_by_triangle_area,
    resample_pyvista_mesh,
)
from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
    scale_to_volume,
)
from irregular_object_packing.mesh.utils import print_mesh_info
from irregular_object_packing.packing import initialize as init
from irregular_object_packing.packing import nlc_optimisation as nlc
from irregular_object_packing.packing import plots
from irregular_object_packing.packing.optimizer_data import (
    IterationData,
    OptimizerData,
    SimConfig,
)


class Optimizer(OptimizerData):

    def __init__(
        self, shape: PolyData, container: PolyData, config: SimConfig, plotter=None, description="default"
    ):
        super().__init__()
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.config = config
        self.cat_data = CatData.default()
        self.tf_arrays = np.empty(0)
        self.plotter: pv.Plotter = plotter
        self.scaling_barrier_list = np.linspace(
            self.config.init_f,
            self.config.final_scale,
            num=self.config.n_scaling_steps + 1,
        )[1:]
        self.description = description
        self.executor = PoolExecutor()
        self.pbar1 = None
        self.pbar2 = None
        self.pbar3 = None
        self.log = logging.getLogger(__name__)
        self.log.setLevel(config.log_lvl)

    @property
    def curr_max_scale(self):
        return self.scaling_barrier_list[self.i_b]

    @ property
    def start_scale(self):
        return self.scaling_barrier_list[self.i_b - 1] if self.i_b > 0 else self.config.init_f

    # ----------------------------------------------------------------------------------------------
    # SETUP functions
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def from_state(statefile: str) -> 'Optimizer':
        state = OptimizerData.load_state(statefile)
        optimizer = Optimizer(
            shape=state.shape0,
            container=state.container0,
            config=state.settings,
        )
        optimizer.description = state.description
        optimizer.tf_arrays = state.tf_arrays
        optimizer.i = 0
        optimizer.i_b = state.iteration_data.i_b
        optimizer.curr_sample_rate = state.iteration_data.sample_rate
        return optimizer

    def setup(self):
        # init current state
        self.curr_sample_rate = self.shape0.n_faces
        object_coords, skipped = init.init_coordinates(
            container=self.container0,
            mesh=self.shape0,
            coverage_rate=self.config.r,
            f_init=self.config.init_f,
        )
        n_objects = len(object_coords)
        # self.resample_meshes(self.settings.init_f)
        self.log.debug(
            f"Skipped {skipped} points to avoid overlap with container")
        self.log.info(f"Setup with settings: \n{self.config}")
        self.log.info(f"Number of objects: {n_objects}")

        object_rotations = np.random.uniform(-np.pi, np.pi, (n_objects, 3))

        # SET TRANSFORM DATA
        self.tf_arrays = np.empty((n_objects, 7))

        for i in range(n_objects):
            tf_arr_i = np.array(
                [self.config.init_f, *object_rotations[i], *object_coords[i]]
            )
            self.tf_arrays[i] = tf_arr_i

        self.update_data(-1, -1)
        objects = self.current_meshes(shape=self.shape0)
        overlaps = compute_object_collisions(objects)
        if len(overlaps) > 0:
            raise ValueError(
                f"Initial object placements show overlaps for {overlaps}"
            )

        overlaps = compute_container_violations(objects, self.container0)
        if len(overlaps) > 0:
            raise ValueError(
                "Initial object placements show container violations for"
                f" {overlaps} objects."
            )

        if self.plotter is not None:
            self.plotter.show(interactive=True, interactive_update=True)

    def setup_pbars(self):
        self.pbar1 = tqdm(
            range(self.config.n_scaling_steps), desc="scaling\t", position=0, leave=True, initial=self.i_b
        )
        self.pbar2 = tqdm(range(self.config.itn_max), desc="Iteration\t", position=1, leave=True, initial=self.i)
        self.pbar3 = tqdm(range(self.n_objs), desc="Object\t", position=2, leave=True, initial=0)

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    def update_data(self, i_b, i, viol_data=()):
        self.log.info(f"Updating data for {i_b=}, {i=}")
        iterdata = IterationData(
            i,
            i_b,
            self.scaling_barrier_list[i_b - 1] if i_b > 0 else self.config.init_f,
            self.scaling_barrier_list[i_b],
            np.count_nonzero(self.tf_arrays[:, 0] >= self.scaling_barrier_list[i_b]),
            self.curr_sample_rate,
            *viol_data
        )
        self.add(self.tf_arrays, self.cat_data, iterdata)

    # def log(self, msg, log_lvl=LOG_LVL_INFO):
    #     if log_lvl > self.config.log_lvl:
    #         return

    #     msg = LOG_PREFIX[log_lvl] + msg + f"[i={self.idx}]"
    #     if self.pbar1 is None:
    #         print(msg)
    #     else:
    #         self.pbar1.write(msg)

    def report(self):
        df = pd.DataFrame(
            data=self.tf_arrays, columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"]
        )
        return df

    def sample_rate_mesh(self, scale_factor):
        if self.config.dynamic_simplification:
            return int(mesh_simplification_condition(scale_factor, self.config.alpha, self.config.beta) * self.shape0.n_faces * self.config.upscale_factor)
        return self.config.sample_rate  # currently simple

    def resample_meshes(self, scale_factor=None):
        self.log.info("resampling meshes")
        if scale_factor is None:
            scale_factor = self.curr_max_scale

        if self.config.sample_rate is not None:
            self.curr_sample_rate = self.sample_rate_mesh(scale_factor)
            self.shape = resample_pyvista_mesh(self.shape0, self.curr_sample_rate)
            self.container = resample_mesh_by_triangle_area(self.shape, self.container0)
        assert self.shape.is_manifold
        assert self.container.is_manifold
        self.log.info(f"container: n_faces: {self.container.n_faces}[sampled]/{self.container0.n_faces}[original]")
        self.log.info(f"mesh: n_faces: {self.curr_sample_rate}[sampled]/{self.shape0.n_faces}[original]")

    def run(self, start_idx=None, end_idx=None, Ni=-1):
        try:
            self.executor = PoolExecutor(thread_name_prefix="optimizer")
            self._run(start_idx, end_idx, Ni)
            self.executor.shutdown(wait=False, cancel_futures=False)
        except KeyboardInterrupt:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.write_state()
        self.log.info("Exiting optimizer.run()")

    def _run(self, start_idx=None, end_idx=None, Ni=-1):
        self.setup_pbars()
        if start_idx is None:
            start_idx = self.i_b
        if end_idx is None:
            end_idx = self.config.n_scaling_steps

        for i_b in range(start_idx, end_idx):
            self.i_b = i_b
            self.resample_meshes(self.curr_max_scale)
            self.log.info(f"Starting scaling step {i_b}")
            self.pbar1.set_postfix(ƒ_max=f"{self.curr_max_scale:.3f}")
            self.pbar2.reset()

            for i in range(self.config.itn_max):
                self.i = i
                self.log.info(f"Starting iteration [{i}, scale_step:{i_b}] total: {self.idx}")
                self.pbar3.reset()
                self.pbar3.set_postfix(total=self.idx)
                if self.iteration() is False:
                    continue

                # administrative stuff
                # self.process_iteration()
                self.pbar2.update()

                if Ni != -1 and self.idx >= Ni:
                    return

                if self.step_should_terminate():
                    break


            self.pbar1.update()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def iteration(self):
        """Perform a single iteration of the optimisation."""

        # DOWN SAMPLE MESHES
        try:
            self.normals, self.cat_cells = self.compute_cat_cells()
        except RuntimeError as e:
            self.log.debug(f"RuntimeError: {e}")
            self.log.debug("Scaling down and trying again...")
            for i in range(self.n_objs):
                self.reduce_scale(i, scale=0.99)
            return False

        # Check the quality if the cat cells
        Points_without_faces = []
        for normals in self.normals:
            if len(normals) == 0:
                Points_without_faces.append(normals)

        if len(Points_without_faces) > 0:
            self.log.warning(f"there are {len(Points_without_faces)} points without faces. ids: {Points_without_faces}")

        # GROWTH-BASED OPTIMISATION
        self.log.info("optimizing cells...")
        if self.config.sequential is False:
            self.parallel_optimisation()
        else:
            for obj_id, previous_tf_array in enumerate(self.tf_arrays):
                self.tf_arrays[obj_id] = self.local_optimisation(obj_id, previous_tf_array, self.curr_max_scale)

        return True




    def parallel_optimisation(self):
        tasks = []
        for obj_id, previous_tf_array in enumerate(self.tf_arrays):
            task = self.executor.submit(self.parallel_local_optimisation, obj_id, previous_tf_array, self.curr_max_scale)
            tasks.append(task)
        wait(tasks)

    """workaround for setting the tf_arrays in parallel"""
    def parallel_local_optimisation(self,obj_id, previous_tf_array, max_scale):
        self.tf_arrays[obj_id] = self.local_optimisation(obj_id, previous_tf_array, max_scale)

    def local_optimisation(self, obj_id, previous_tf_array, max_scale):
        # get all the vertices for the points of this object
        last_ids = [-1]
        for i in range(self.n_objs):
            last_ids.append(last_ids[-1] + self.objects[i].n_points)

        vertices = self.objects[obj_id].points

        # get the points for
        p_first, p_last = last_ids[obj_id]+1, last_ids[obj_id + 1]
        face_normals = self.normals[p_first:p_last]

        new_tf = nlc.compute_optimal_growth(
            previous_tf_array=previous_tf_array,
            obj_coord=self.object_coords[obj_id],
            vertices=vertices,
            face_normals=face_normals,
            max_scale=max_scale,
            scale_bound=(self.config.init_f, None),
            max_angle=self.config.max_a,
            max_t=self.config.max_t * max_scale if self.config.max_t is not None else None,
            padding=self.config.padding,
        )
        self.pbar3.update()
        return new_tf

    def compute_cat_cells(self, kwargs=None, new=True) -> tuple[np.ndarray, np.ndarray]:
        self.log.info("Computing CAT cells")
        if kwargs is None:
            kwargs = {
                # "nobisect": True,
                "steinerleft": 0,
                "minratio": 10.0,
                # "cdt": 1,
                "quality": False,
                "opt_scheme": 0,
                "switches": "O/0",
                # "verbose": 2,
                "quiet": True,
            }

        # TRANSFORM MESHES TO OBJECT COORDINATES, SCALE, ROTATION
        self.objects = self.current_meshes()

        # Compute the CDT
        tetmesh = cat.compute_cdt(self.objects + [self.container], kwargs)

        # The point sets are sets(uniques) of tuples (x,y,z) for each object, for quick lookup
        obj_point_sets = [set(map(tuple, obj.points)) for obj in self.objects] + [
            set(map(tuple, self.container.points))
        ]

        # Check that all points are accounted for
        assert np.sum([len(obj) for obj in obj_point_sets]) == np.sum([obj.n_points for obj in self.objects] + [self.container.n_points])
        assert np.sum([len(obj) for obj in obj_point_sets]) == tetmesh.n_points, "Some points are created by tetmesh"

        # COMPUTE CAT CELLS
        return cat.compute_cat_faces_new(
                tetmesh, obj_point_sets, self.tf_arrays[:, 4:]
            )


    def step_should_terminate(self):
        """Returns true if all objects are scaled to the current max scale."""

        are_scaled = [arr[0] >= self.curr_max_scale for arr in self.tf_arrays]
        count = 0
        for i in range(self.n_objs):
            if are_scaled[i]:
                count += 1
                self.log.debug(f"Object {i} has reached the scaling barrier")

        self.log.info(f"{count}/{self.n_objs} objects have reached the scaling barrier")
        self.log.info(f"scales: {[f'{f[0]:.2f}' for f in self.tf_arrays]}")
        if count == self.n_objs:
            return True
        return False

    # ----------------------------------------------------------------------------------------------
    # VALIDITY CHECKS
    # ----------------------------------------------------------------------------------------------
    def process_iteration(self):
        i, ib = self.i, self.i_b
        is_correct = False
        while is_correct is False:
            p_meshes = self.current_meshes()
            cat_meshes = self.final_cat_meshes()
            cat_viols, con_viols, collisions = compute_all_collisions(p_meshes, cat_meshes, self.container, set_contacts=False)
            self.log_violations((cat_viols, con_viols, collisions))
            violating_ids = set()
            for ((obj_ida, obj_idb), _) in collisions:
                violating_ids.add(obj_ida)
                violating_ids.add(obj_idb)

            for (obj_id, _) in con_viols:
                violating_ids.add(obj_id)

            is_correct = len(violating_ids) == 0
            if len(violating_ids) != 0:
                self.log.info("reducing scale for violating objects: " + str(violating_ids))
                for id in violating_ids:
                    self.reduce_scale(id, scale=0.93)

        self.update_data(ib, i, (cat_viols, con_viols, collisions))

    def reduce_scale(self, id, scale=0.95):
        self.tf_arrays[id][0] *= scale

    def check_closed_cells(self):
        cat_cells = [
            PolyData(*cat.catdatacell_to_points_and_faces(self.cat_data, obj_id))
            for obj_id in range(self.n_objs)
        ]
        for i, cell in enumerate(cat_cells):
            if not cell.is_manifold:
                self.log.error(
                    f"CAT cell of object {i} is not manifold"
                )

    def log_violations(self, violations):
        if len(violations[0]) > 0:
            self.log.warning(f"! cat violation found {violations[0]}")
        if len(violations[1]) > 0:
            self.log.warning(f"! container violation found {violations[1]}")
        if len(violations[2]) > 0:
            self.log.warning(f"! collisions found {violations[2]}")
        sleep(0.5)  # for easier spotting in the terminal

    def store_state(self, meshes, name=""):
        sum = pv.PolyData()
        for mesh in meshes:
            sum = sum + mesh

        sum.save(f"../dump/{name}error-{time():.0f}.stl", sum)

    @staticmethod
    def default_setup() -> "Optimizer":
        DATA_FOLDER = "./../../data/mesh/"

        mesh_volume = 0.2
        container_volume = 10

        loaded_mesh = pv.read(DATA_FOLDER + "RBC_normal.stl")
        container = pv.Sphere()

        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
        print_mesh_info(original_mesh, "original mesh")

        config = SimConfig(
            itn_max=100,
            n_scaling_steps=9,
            r=0.3,
            final_scale=1.0,
            log_lvl=logging.INFO,
            init_f=0.1,
            max_t=mesh_volume**(1 / 3) * 2,
            padding=1E-4 * mesh_volume**(1 / 3),
            dynamic_simplification=True,
            alpha=0.1,
            beta=0.5,
            upscale_factor=1,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, config, plotter, description="cells_in_sphere")
        return optimizer

    @staticmethod
    def simple_shapes_setup() -> "Optimizer":
        mesh_volume = 0.2

        container_volume = 10

        original_mesh = pv.Cube().triangulate().extract_surface()
        container = pv.Cube().triangulate().extract_surface()

        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(original_mesh, mesh_volume)
        print_mesh_info(original_mesh, "original mesh")

        settings = SimConfig(
            itn_max=100,
            n_scaling_steps=9,
            r=0.3,
            final_scale=1,
            sample_rate=None,
            log_lvl=logging.WARNING,
            init_f=0.1,
            # padding=0,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, settings, plotter)
        return optimizer


# %%
# TODO: Refactor the OPTIMIZER DATA CLASS
# TODO: Refactor the optimizer class functions:
#   - process_iteration
#   - check_closed_cells
#   - log_violations
#   - store_state
#   - reduce_scale
#   - update_data

optimizer = Optimizer.default_setup()
# optimizer = Optimizer.simple_shapes_setup()
optimizer.setup()
optimizer.config.sequential = False
#%%
optimizer.run(Ni=1)
# optimizer.run()
# %%
# optimizer.compute_cat_cells(kwargs={
kwargs={
    # "nobisect": True,
    "minratio": 10.0,
    "quality": False,
    "opt_scheme": 0,
    "verbose": 2,
    "quiet": False,
    # "steinerleft": 0, # switch: S
    # "cdt": 1,  # switch: D
    # "opt_scheme": 0,  # switch: O/#
    "switches": "O/0DS0",
}
object_meshes = optimizer.current_meshes()

# Compute the CDT
tetmesh = cat.compute_cdt(object_meshes + [optimizer.container], kwargs)

# The point sets are sets(uniques) of tuples (x,y,z) for each object, for quick lookup
obj_point_sets = [set(map(tuple, obj.points)) for obj in object_meshes] + [
    set(map(tuple, optimizer.container.points))
]
objects_npoints = [len(obj) for obj in obj_point_sets]
cells = get_cell_arrays(tetmesh.cells)
rel_cells, _ = filter_relevant_cells(cells, objects_npoints)
data = cat.compute_cat_faces(tetmesh, obj_point_sets, optimizer.object_coords)

#%%
obj_id = 10
faceless_points = data.get_faceless_points_for(obj_id)
vertices = tetmesh.points[faceless_points]
print(vertices)

# get all the cells from the relevant cells that are part of the object
object_cell_ids = [x.id for x in filter(lambda x: x.belongs_to_obj(obj_id), rel_cells)]
object_cells = tetmesh.extract_cells(object_cell_ids)

faceless_cell_ids = tetmesh.find_containing_cell(vertices)
faceless_cells = tetmesh.extract_cells(faceless_cell_ids)
cat_cell = PolyData(*cat.catdatacell_to_points_and_faces(data, obj_id))

plotter = pv.Plotter()
pv_points = pv.PolyData(vertices)
# plotter.add_mesh_clip_plane(tetmesh, color="w", opacity=0.2, show_edges=True)
def plane_func(normal, origin):
    cat_clip = cat_cell.clip(normal=normal, origin=origin, crinkle=True)
    tet_clip = object_cells.clip(normal=normal, origin=origin, crinkle=True)
    plotter.add_mesh(tet_clip, color="w", opacity=0.6, show_edges=True)
    plotter.add_mesh(cat_clip, color="y", opacity=0.6, show_edges=True)

plotter.add_mesh(pv_points, color="w", opacity=1, show_vertices=True, point_size=10)
plotter.add_mesh(faceless_cells, color="r", opacity=0.9, show_edges=True)
plotter.add_mesh(object_meshes[obj_id], color="b", opacity=0.7, show_edges=True)
plotter.add_plane_widget(plane_func)
# plotter.add_mesh(cat_cell, color="y", opacity=0.7, show_edges=True)
plotter.show()

print(f"cat_cell is manifold: {cat_cell.is_manifold}")

#%%
# cat.compute_cat_faces_new(tetmesh, obj_point_sets, optimizer.tf_arrays[:, 4:])
# %%
# state_file = "state-cells_in_sphere-n15_cv10.0_f0.7000000000000001.pickle"
# state_file = "state-cells_in_sphere-n15_cv10.0_f1.0-t1683810762.pickle"
# optimizer = Optimizer.from_state(state_file)
# %load_ext pyinstrument
# %%
optimizer.run(Ni=1)
# optimizer.run()


# %%

# reload(plots)
save_path = f"../dump/full_growth_{optimizer.n_objs}_cells_{time()}"
# plots.generate_gif(optimizer , save_path + ".gif")

# reload(plots)


def plot_step(optimizer: Optimizer, step, meshes, cat_meshes, container):
    plotter = pv.Plotter()
    plots.plot_simulation_scene(plotter, meshes, cat_meshes, container, c_kwargs={"show_edges": False, "edge_color": "purple"})
    plotter.add_text(optimizer.status(step).table_str, position="upper_left")
    plotter.show()
    return plotter


step = optimizer.idx
meshes_before, meshes_after, cat_meshes, container = optimizer.recreate_scene(step)
plotter = plot_step(optimizer, step, meshes_after, cat_meshes, container)
# plotter.save_graphic(f"{save_path}.pdf")
# %%


obj_i = 9
plotter = plots.plot_step_single(
    meshes_after[obj_i], cat_meshes[obj_i],  # container=container,
    # meshes_after[obj_i], cat_meshes[obj_i],  # container=container,
    cat_opacity=0.6, mesh_opacity=1 , clipped=True, title="cat violation",
    # other_meshs=[meshes_after[1], ],
    # tetmesh=tetmesh,
    # c_kwargs={"show_edges": True, "edge_color": "purple", "show_vertices": True, "point_size": 10},
    m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
    cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
    oms_kwargs=[
        {"show_edges": True, "color": "w", "edge_color": "red", "show_vertices": True, "point_size": 1, }
    ],
)

# plotter.add_point_scalar_labels(optimizer.cat_data.get_empty_faces(obj_i), point_size=10, font_size=10, point_color="red", labels="test")
# for obj_i in range(len(meshes_before)):
# points_outside = compute_outside_points(cat_meshes[obj_i], meshes_after[obj_i])
# plotter.add_points(points_outside, color="red", point_size=10, style="render_points_as_spheres")
# %%
# plotter = plots.plot_step_single(
#     # meshes_before[obj_i], cat_meshes[obj_i],  # container=container,
#     meshes_after[obj_i], cat_meshes[obj_i],  # container=container,
#     cat_opacity=0.7, mesh_opacity=1 , clipped=True, title="cat overlap and collision",
#     other_meshs=[meshes_after[2], ],
#     # tetmesh=tetmesh,
#     # c_kwargs={"show_edges": True, "edge_color": "purple", "show_vertices": True, "point_size": 10},
#     m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
#     cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
#     oms_kwargs=[
#         {"show_edges": True, "color": "w", "edge_color": "red", "show_vertices": True, "point_size": 1, }
#     ],
# )

# # %%
# # # store cat mesh in file

# title = "NLC_opt_too_much"
# obj_ids, step = [10, 13], 89


# def store_issue_files(optimizer, step, title, obj_ids):
#     issue_name = f"issue{title}_{int(time())}"
#     folder_dir = f"../dump/issue_reports/{issue_name}/"
#     mkdir(folder_dir)

#     meshes_before, meshes_after, cat_meshes, container = optimizer.recreate_scene(step)
#     tetmesh, filtered_tetmesh, _ = optimizer.reconstruct_delaunay(step)
#     for obj_i in obj_ids:
#         cat_meshes[obj_i].save(folder_dir + f"cat[o{obj_i}i{step}].stl")
#         meshes_before[obj_i].save(folder_dir + f"obj_before[o{obj_i}i{step}].stl")
#         meshes_after[obj_i].save(folder_dir + f"obj_after[o{obj_i}i{step}].stl")

#     tetmesh, filtered_tetmesh, _ = optimizer.reconstruct_delaunay(step)
#     tetmesh.save(folder_dir + f"tetmesh[i{step}].vtk")
#     filtered_tetmesh.save(folder_dir + f"filtered_tetmesh[i{step}].vtk")


# store_issue_files(optimizer, step, title, obj_ids)

# # %%


# @cprofile
# def profile_optimizer():
#     optimizer.run(start_idx=optimizer.i_b, Ni=1)


# profile_optimizer()
# # # %%


# # # %%


# # fig, ax = plt.subplots()

# # a = [0.05, 0.15, 0.25]
# # b = [0.1, 0.2, 0.3, 0.5]

# # x = np.linspace(0, 1, 100)

# # for ai in a:
# #     for bi in b:
# #         print(f"{ai} {bi}`")
# #         ax.plot(mesh_simplification_condition(x, ai, bi), label=f"a:{ai:.2f},  b:{bi:.2f}")
# # ax.legend()

# # # %%

# # %%

# %%
