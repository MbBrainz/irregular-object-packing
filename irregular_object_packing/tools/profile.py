# from line_profiler import LineProfiler
import cProfile
import profile

# visualise using gprof2dot:
# gprof2dot.py -f pstats output.pstats | dot -Tpng -o dump/output.png


# def lineprofile(func):
#     def inner(*args, **kwargs):
#         profiler = LineProfiler()
#         profiler.add_function(func)
#         profiler.enable_by_count()
#         return func(*args, **kwargs)
#     return inner


def cprofile(func):
    """
    Decorator (function wrapper) that profiles a single function
    @profileit()
    def func1(...)
        # do something
        pass
    """

    def inner(*args, **kwargs):
        func_name = func.__name__ + "-c.pstats"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(func_name)
        return retval

    return inner


# # Example use 
# @pprofile
# def prof_face_coord_to_points_and_faces():
#     return face_coord_to_points_and_faces(cat_cells[0])
# cat_points, poly_faces = prof_face_coord_to_points_and_faces()

def pprofile(func):
    """
    Decorator (function wrapper) that profiles a single function
    @profile()
    def func1(...)
            # do something
        pass
    """

    def inner(*args, **kwargs):
        func_name = func.__name__ + ".pstats"
        prof = profile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(func_name)
        return retval

    return inner
