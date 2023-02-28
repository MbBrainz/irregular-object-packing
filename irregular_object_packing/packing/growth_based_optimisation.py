# %%
import numpy as np
from scipy.optimize import minimize

from irregular_object_packing.packing import nlc_optimisation as nlc
from irregular_object_packing.packing import packing as pkn


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
