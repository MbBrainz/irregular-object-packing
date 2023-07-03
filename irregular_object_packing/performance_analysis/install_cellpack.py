import os

from irregular_object_packing.tools.utils import get_project_root

CELLPACK_PATH = "HemoCell/tools/packCells"
CELLPACK_EXECUTABLE = f"./{CELLPACK_PATH}/build/packCells"
PERFORMANCE_ANALYSIS_PATH = str(get_project_root()) + "/irregular_object_packing/performance_analysis"

def check_path():
    current_path = os.getcwd()
    if current_path != PERFORMANCE_ANALYSIS_PATH:
        raise ValueError(f"executed from {current_path} instead of the performance analysis dir{PERFORMANCE_ANALYSIS_PATH}. ")
    return

def clone_and_compile_cellpack():
    os.system("git clone https://github.com/UvaCsl/HemoCell.git")
    os.chdir(CELLPACK_PATH)
    os.system("mkdir build && cd build")
    os.system("cmake ..")
    os.system("make")
    os.chdir(PERFORMANCE_ANALYSIS_PATH)

    return os.getcwd()

def run_cellpack(flags: str = "-h"):
    check_path()
    os.system(f"{CELLPACK_EXECUTABLE} {flags}")
    return

if __name__ == "__main__":
    check_path()
    clone_and_compile_cellpack()
    run_cellpack()


