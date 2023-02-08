from screeninfo import get_monitors

def get_offset_to_second_monitor():
    """If there is only monitor returns (0, 0). Else, returns offset to second monitor.
    Returns:
        Tuple[int, int] Offset to the second monitor coordinates.
    """
    max_resolution = None
    offset_coordinates = None
    for m in get_monitors():
        resolution = m.width * m.height
        if max_resolution is None or resolution > max_resolution:
            offset_coordinates = (m.x, m.y, m.width, m.height)            
    return offset_coordinates


if __name__ == "__main__":
    print(get_offset_to_second_monitor())