from PIL import Image, ImageDraw
import numpy as np


def generate_mask_color(sprites, image_mode):
    labels = sprites.keys()
    is_rgba = image_mode == "RGBA"
    colors = []
    num_of_colors = 0
    while num_of_colors < len(labels):
        color = [
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)]

        if is_rgba:
            color.append(np.random.randint(200, 255))

        color = tuple(color)

        if color not in colors:
            colors.append(color)
            num_of_colors += 1
    return dict(zip(labels, colors))


def create_sprite_labels_image(image, sprites, label_map, background_color=(255, 255, 255)):
    image_mode = "RGBA" if len(background_color) == 4 else "RGB"
    height = image.height
    width = image.width
    color_dict = generate_mask_color(sprites, image_mode)
    image = Image.new(image_mode, (width, height), background_color)
    draw_context = ImageDraw.Draw(image, image_mode)
    for row in range(height):
        for col in range(width):
            label = label_map[row][col]
            if label:
                try:
                    draw_context.point(
                        (col, row), color_dict[label])
                except KeyError:
                    pass

    for label, sprite in sprites.items():
        try:
            draw_context.rectangle(
                [sprite.top_left, sprite.bottom_right],
                outline=color_dict[label])
        except KeyError:
            pass

    del draw_context
    return image
