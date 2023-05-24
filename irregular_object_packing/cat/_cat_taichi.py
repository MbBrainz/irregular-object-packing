import taichi as ti


@ti.data_oriented
class CatTaichi():
    def __init__(self) -> None:
        pass


@ti.data_oriented
class ObjData():
    def __init__(self, mesh) -> None:
        pass

# Like the parameters you see in the mesh examples like velocity or weight, we could add an index to each vertex with the related object id.
# maybe we could also add a initially empty vector/list there to store the cat face lateron, what do you think?
# for each cell we could add a type of ([1122], [3331], [4444], [1111]) to conveniently branch the computation of cat faces.
