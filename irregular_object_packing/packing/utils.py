# %%

def print_transform_array(array):
    symbols = ["f", "θ_x", "θ_y", "θ_z", "t_x", "t_y", "t_z"]
    header = " ".join([f"{symbol+':':<8}" for symbol in symbols])
    row = " ".join([f"{value:<8.3f}" for value in array])
    print(header)
    print(row + "\n")

def log_violations(logger,idx, violations ):
    if len(violations[0]) > 0:
        logger.warning(f"[i {idx}]! cat violation found {violations[0]}")
    if len(violations[1]) > 0:
        logger.warning(f"[{idx}]! container violation found {violations[1]}")
    if len(violations[2]) > 0:
        logger.warning(f"[{idx}]! collisions found {violations[2]}")
