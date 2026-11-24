"""Render the original WSI viewport at its original pyramid level."""
import math
from PIL import Image


class ViewportPixelLimitError(ValueError):
    """The requested read region exceeds the configured pixel limit."""


def viewport_read_spec(slide, viewport, output_size):
    x, y, width, height = (float(viewport[key]) for key in ('x', 'y', 'w', 'h'))
    level = slide.get_best_level_for_downsample(
        max(width / output_size[0], height / output_size[1], 1.0))
    downsample = float(slide.level_downsamples[level])
    read_size = (max(1, math.ceil(width / downsample)), max(1, math.ceil(height / downsample)))
    scale = min(output_size[0] / read_size[0], output_size[1] / read_size[1])
    resized = tuple(max(1, round(size * scale)) for size in read_size)
    origin_x, origin_y = math.floor(x), math.floor(y)
    left, top = max(0, origin_x), max(0, origin_y)
    right = min(slide.dimensions[0], origin_x + read_size[0] * downsample)
    bottom = min(slide.dimensions[1], origin_y + read_size[1] * downsample)
    crop_size = ((math.ceil((right - left) / downsample), math.ceil((bottom - top) / downsample))
                 if right > left and bottom > top else (0, 0))
    return {'level': level, 'downsample': downsample, 'read_size': read_size,
            'scale': scale, 'resized': resized, 'origin': (origin_x, origin_y),
            'read_origin': (left, top), 'crop_size': crop_size,
            'pixels': crop_size[0] * crop_size[1]}


def render_wsi_viewport(slide, viewport, output_size, max_read_pixels=None):
    spec = viewport_read_spec(slide, viewport, output_size)
    if max_read_pixels is not None and spec['pixels'] > max_read_pixels:
        raise ViewportPixelLimitError(
            f"level={spec['level']}, read_size={spec['crop_size']}, pixels={spec['pixels']} exceeds {max_read_pixels}")
    canvas = Image.new('RGB', spec['resized'], 'black')
    if spec['pixels'] == 0:
        return canvas
    rgba = slide.read_region(spec['read_origin'], spec['level'], spec['crop_size']).convert('RGBA')
    image = Image.new('RGB', rgba.size, 'black')
    image.paste(rgba, mask=rgba.getchannel('A'))
    crop_resized = tuple(max(1, round(size * spec['scale'])) for size in spec['crop_size'])
    image = image.resize(crop_resized, Image.Resampling.LANCZOS)
    position = tuple(round((read - origin) / spec['downsample'] * spec['scale'])
                     for read, origin in zip(spec['read_origin'], spec['origin']))
    canvas.paste(image, position)
    return canvas
