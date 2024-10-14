import numpy as np
from PIL import Image, ImageDraw


def draw_egopath(img, egopath, opacity=0.5, color=(0, 189, 80), crop_coords=None):
    """Overlays the train ego-path on the input image.

    Args:
        img (PIL.Image.Image): Input image on which rails are to be visualized.
        egopath (list or numpy.ndarray): Ego-path to be visualized on the image, either as a list of points (classification/regression) or as a mask (segmentation).
        opacity (float, optional): Opacity level of the overlay. Defaults to 0.5.
        color (tuple, optional): Color of the overlay. Defaults to (0, 189, 80).
        crop_coords (tuple, optional): Crop coordinates used during inference. If provided, a red rectangle will be drawn around the cropped region. Defaults to None.

    Returns:
        PIL.Image.Image: Image with the ego-path overlay.
    """
    vis = img.copy()
    if isinstance(egopath, list):  # classification/regression
        left_rail, right_rail = egopath
        if not left_rail or not right_rail:
            return vis
        points = left_rail + right_rail[::-1]
        mask = Image.new("RGBA", vis.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask)
        draw.polygon([tuple(xy) for xy in points], fill=color + (int(255 * opacity),))
        vis.paste(mask, (0, 0), mask)
    elif isinstance(egopath, Image.Image):  # segmentation
        mask = Image.fromarray(np.array(egopath) * opacity).convert("L")
        colored_mask = Image.new("RGBA", mask.size, color + (0,))
        colored_mask.putalpha(mask)
        vis.paste(colored_mask, (0, 0), colored_mask)
    if crop_coords is not None:
        draw = ImageDraw.Draw(vis)
        draw.rectangle(crop_coords, outline=(255, 0, 0), width=1)
    return vis

def drawAnnotation(image, annotation):
    """Overlays the Ground Truth on the input image.

    Args:
        image (PIL.Image.Image): Input image on which rails are to be visualized.
        annotation (list): Raw annotation of an image that should be visualized.

    Returns:
        PIL.Image.Image: Image with the ego-path overlay.
    """

    left_rail_color = (255, 0, 0)   # Rot für die linke Schiene
    right_rail_color = (0, 0, 255)  # Blau für die rechte Schiene
    point_radius = 3                # Punktgröße

    draw = ImageDraw.Draw(image)

    # Linke Schiene einzeichnen (falls vorhanden)
    if "left_rail" in annotation:
        left_rail_coords = annotation["left_rail"]
        for i, (x, y) in enumerate(left_rail_coords):
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=left_rail_color)
            # Linien zwischen den Punkten zeichnen
            if i > 0:
                prev_x, prev_y = left_rail_coords[i - 1]
                draw.line([(prev_x, prev_y), (x, y)], fill=left_rail_color, width=2)
    
    # Rechte Schiene einzeichnen (falls vorhanden)
    if "right_rail" in annotation:
        right_rail_coords = annotation["right_rail"]
        for i, (x, y) in enumerate(right_rail_coords):
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=right_rail_color)
            # Linien zwischen den Punkten zeichnen
            if i > 0:
                prev_x, prev_y = right_rail_coords[i - 1]
                draw.line([(prev_x, prev_y), (x, y)], fill=right_rail_color, width=2)
    
    return image  # Das annotierte Bild zurückgeben