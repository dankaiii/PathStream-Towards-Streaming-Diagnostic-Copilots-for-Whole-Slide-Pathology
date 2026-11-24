"""Valid-image fractions for diagnosis evidence tokens."""

def area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def token_weights(frame, grid_height, grid_width):
    """Fraction of each merged visual token cell covered by valid requested pixels."""
    render = frame["render"]
    rect = render["valid_pixel_rect"]
    ow, oh = render["output_size"]
    dx, dy = ow / grid_width, oh / grid_height
    return [area(intersection(rect, [x * dx, y * dy, (x + 1) * dx, (y + 1) * dy])) / (dx * dy)
            for y in range(grid_height) for x in range(grid_width)]
