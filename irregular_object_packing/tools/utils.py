from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


if __name__ == "__main__":
    print(get_project_root())
