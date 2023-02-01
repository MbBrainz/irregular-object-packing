import taichi as ti


def test_taichi_init():
    ti.init(arch=ti.gpu, kernel_profiler=True)
    print("success")
