# Document function description
#   insert lib path into system path temporarily.
from pathlib import Path
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


cur_path = Path(__file__)
lib_path = cur_path.parent.parent
add_path(str(lib_path))


if __name__ == '__main__':
    print(sys.path)
    