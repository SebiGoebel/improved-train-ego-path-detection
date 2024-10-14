import numpy as np
from PIL import Image, ImageDraw

import json
import os

test_visualizations = True

def draw_coordinates_on_image(coord_list1, coord_list2, image_size):
    # Create a white image
    img = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw pixels from coord_list1 in black
    for coord in coord_list1:
        x, y = coord
        draw.point((x, y), fill=(0, 0, 0))
    
    # Draw pixels from coord_list2 in black
    for coord in coord_list2:
        x, y = coord
        draw.point((x, y), fill=(0, 0, 0))
    
    # Save the image (optional)
    img.save("output_lines_test_image.png")

def find_edge_pixel_per_row(polygon, image_size):
    # Create a mask image with the same size as the input image
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, outline=1, fill=0)  # Draw the polygon edges only

    mask_array = np.array(mask) # Convert the mask to a numpy array
    first_edge_pixels = [] # Initialize a list to store the coordinates of the first edge pixel in each row
    rightmost_edge_pixels = [] # Initialize a list to store the coordinates of the rightmost edge pixel in each row

    if test_visualizations:
        # um umrissen des polygons an zu zeigen
        img = Image.new("RGB", image_size, (255, 255, 255))  # Weißes Hintergrundbild
        draw = ImageDraw.Draw(img)
        draw.polygon(polygon, outline=(0, 0, 0))  # Zeichnet das Polygon (schwarze Linie)
        img.save("polygon_test_img.png") # Speichern des Bildes
        print("saved test polygon img")

    # Iterate over each row (y-coordinate)
    for y in range(image_size[1]):
        # Find the first non-zero (edge) pixel in the current row
        for x in range(image_size[0]):
            if mask_array[y, x] > 0:
                first_edge_pixels.append([x, y])
                break  # Exit the loop once the first edge pixel is found
    
        # Find the last non-zero (edge) pixel in the current row
        for x in range(image_size[0]-1, -1, -1):
            if mask_array[y, x] > 0:
                rightmost_edge_pixels.append([x, y])
                break  # Exit the loop once the rightmost edge pixel is found

    return first_edge_pixels, rightmost_edge_pixels

def write_coordinates_to_json(input_file, left_edge_pixels, right_edge_pixels, output_folder):
    data = {
        input_file: {
            "left_rail": left_edge_pixels,
            "right_rail": right_edge_pixels
        }
    }
    
    # Erstellen des Ausgabeordners, falls er nicht existiert
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Pfad zur JSON-Datei
    json_extension = ".json"
    outname_json = f"{os.path.splitext(os.path.basename(input_file))[0]}{json_extension}"
    json_filepath = os.path.join(output_folder, outname_json)
    
    # Schreiben der Daten in die JSON-Datei
    with open(json_filepath, "w") as json_file:
        #json.dump(data, json_file, indent=4)
        json.dump(data, json_file, separators=(', ', ': '))

def filterCoords(img, egopath, opacity=0.5, color=(0, 189, 80), crop_coords=None, input_file=None, output_folder=None):
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
    lowest_y = None

    if isinstance(egopath, list):  # classification/regression
        left_rail, right_rail = egopath
        if not left_rail or not right_rail:
            return vis
        points = left_rail + right_rail[::-1]

        # Find the highest y-value in the polygon
        lowest_y = min(point[1] for point in points)

        # Extract polygon points for find_first_edge_pixel_per_row function
        polygon_points = [tuple(xy) for xy in points]
        # Find the coordinates of the edge pixels in each row
        left_edge_pixels, right_edge_pixels = find_edge_pixel_per_row(polygon_points, vis.size)

        if test_visualizations:
            # visualisierung zum testen
            draw_coordinates_on_image(left_edge_pixels, right_edge_pixels, vis.size)
            print("saved test line img")
        
        if input_file == None:
            print("Error: input_file is not set !!!")
            print("Changing input_file to default_input_file")
            input_file = "default_input_file"

        if output_folder == None:
            print("Error: output_folder is not set !!!")
            print("Changing foldername to default_output_folder")
            output_folder = "default_output_folder"

        write_coordinates_to_json(input_file, left_edge_pixels, right_edge_pixels, output_folder)

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

    # Print the highest y-value of the polygon if found
    if lowest_y is not None:
        print(f"Highest y-value of the polygon: {1080-lowest_y}")
        draw.line((0, lowest_y, vis.width, lowest_y), fill=(255, 0, 0), width=2)
    
    return vis
