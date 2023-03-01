# %%
import numpy as np
from scipy.optimize import minimize

from irregular_object_packing.packing import nlc_optimisation as nlc
from irregular_object_packing.packing import packing as pkn
import irregular_object_packing.packing.chordal_axis_transform as cat


# %%
# nlc.constraint_multiple_points(x, v, facets_sets)
#
# %% [markdown]
# ## Combining CAT cells and NLC optimisation
# We now have a working implementation of the cat cells computation and the NLC optimisation
# The CAT cells are in the form of a dictionary with the key being the object index
# and the value being a dictionary with the key being the vertex
# tuple and the value being the list of facets related to that vertex
# the dict will look as follows for one object:
# ```
# {
#     0: {
#         "all": ["face1", "face2", "face3", "face4", "face5", "face6"],
#         (0, 1, 2): ["face1", "face2", "face3"],
#         (0, 1, 3): ["face1", "face2", "face4"],
#         (0, 2, 3): ["face1", "face5", "face6"],
#     }
# }
# ```
# Each face will be represented by a np.ndarray of 3 or more points.

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### NLC optimisation with CAT cells single interation
# We will now combine the NLC optimisation with the CAT cells to create a single iteration of the optimisation.
#
# %%
# all_k_faces = cat_cells[k].pop("all")
k = 0
irop_data = cat.cat.IropData([])


def optimal_transform(k, irop_data, scale_bound=(0.1, None), max_angle=1 / 12 * np.pi, max_t=None):
    r_bound = (-max_angle, max_angle)
    t_bound = (0, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    constraint_dict = {
        "type": "ineq",
        "fun": nlc.constraints_from_dict,
        "args": (
            k,
            irop_data,
        ),
    }

    res = minimize(nlc.objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    return res.x


tf_arr = optimal_transform(k, irop_data)
