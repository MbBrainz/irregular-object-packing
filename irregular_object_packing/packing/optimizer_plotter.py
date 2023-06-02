from pyvista import Plotter as PvPlotter

from irregular_object_packing.packing import plots
from irregular_object_packing.packing.optimizer_data import OptimizerData


class ScenePlotter:
    def __init__(self, data: OptimizerData):
        self.data = data
        self.cached_step = None
        self.meshes_before = None
        self.meshes_after = None
        self.cat_meshes = None
        self.container = None

    def generate_gif(self, save_path, title=""):
        plots.generate_gif(self.data, save_path, title)

    def plot_step(self, step=None, after_scale=True, save_path=None):
        step = self.data.idx if step is None else step

        if step == self.cached_step:
            meshes_before = self.meshes_before
            meshes_after = self.meshes_after
            cat_meshes = self.cat_meshes
            container = self.container
        else:
            meshes_before, meshes_after, cat_meshes, container = self.data.recreate_scene(step)

        plotter = PvPlotter()
        objects = meshes_after if after_scale is True else meshes_before
        plots.plot_simulation_scene(plotter, objects, cat_meshes, container)

        if save_path is not None:
            plotter.save_graphic(save_path, title=f"{self.data.description}step{step}")
        else:
            plotter.show(auto_close=True)

    def plot_step_object(self, step: int, obj_id: int, after_scale=True, save_path=None):
        if step == self.cached_step:
            mesh_before = self.meshes_before[obj_id]
            mesh_after = self.meshes_after[obj_id]
            cat_mesh = self.cat_meshes[obj_id]
        else:
            mesh_before, mesh_after, cat_mesh = self.data.recreate_object_scene(step, obj_id)

        plotter = PvPlotter(shape=(2,1))

        mesh = mesh_after if after_scale is True else mesh_before
        plotter.subplot(0, 0)
        plots.plot_step_single(
            mesh,
            cat_mesh,
            cat_opacity=0.5, mesh_opacity=0.9,
            m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
            cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
            # other_meshs=[meshes_after[1], ],
            # oms_kwargs=[
            #     {"show_edges": True, "color": "w", "edge_color": "red", "show_vertices": True, "point_size": 1, }
            # ],
        )

    def plot_step_object_compare(self, step: int, obj_id: int, save_path=None):
        if step == self.cached_step:
            mesh_before = self.meshes_before[obj_id]
            mesh_after = self.meshes_after[obj_id]
            cat_mesh = self.cat_meshes[obj_id]
        else:
            mesh_before, mesh_after, cat_mesh = self.data.recreate_object_scene(step, obj_id)

        plotter =PvPlotter(shape=(2, 1))
        plotter.subplot(1, 0)
        plots.plot_step_single(
            mesh_before,
            cat_mesh,
            cat_opacity=0.5, mesh_opacity=0.9,
            m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
            cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
            plotter=plotter,
        )

        plotter.subplot(1, 0)
        plots.plot_step_single(
            mesh_after,
            cat_mesh,
            cat_opacity=0.5, mesh_opacity=0.9,
            m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
            cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
            plotter=plotter,
        )

        if save_path is not None:
            plotter.save_graphic(save_path, title=f"{self.data.description}step{step}obj{obj_id}")
        else:
            plotter.show(auto_close=True)
