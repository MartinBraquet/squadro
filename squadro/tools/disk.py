import pickle
from pathlib import Path


def extend_filename(file_path: str | Path, s: str) -> Path:
    """
    Extends the filename of a file with a string.
    :param file_path:
    :param s:
    :return:

    Example:
    >>> extend_filename("path/to/file.txt", "_new")
    'path/to/file_new.txt'

    >>> extend_filename(Path("file.txt"), "_old")
    PosixPath('file_old.txt')
    """
    is_string = isinstance(file_path, str)
    file_path = Path(file_path)
    file_path_new = file_path.with_name(file_path.stem + s + file_path.suffix)
    if is_string:
        file_path_new = str(file_path_new)
    return file_path_new


def pickle_dump(obj, file_path: str | Path):
    """
    Dumps an object to a file using pickle.
    :param obj:
    :param file_path:
    :return:
    """

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(file_path: str | Path):
    """
    Loads an object from a file using pickle.
    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def mkdir(path: str | Path):
    """
    Creates a directory if it doesn't exist.
    :param path:
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def human_readable_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"
