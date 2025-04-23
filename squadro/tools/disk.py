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
