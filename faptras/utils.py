def get_file_name(path: str) -> str:
    """Extracts file name from the path. E.g. for path /user/video/t7.mp4 returns t7

    Args:
        path (str): path to the file

    Returns:
        str: Extracted file name.
    """
    last_slash = path.rfind('/')
    last_dot = path.rfind('.')
    return path[last_slash + 1:last_dot]