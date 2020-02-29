from PIL import Image
import numpy as np
import exifread
import math
from spriteutils import *
from scipy.spatial import distance


def load_image_and_correct_orientation(file_path_name):
    """
    This function will load image and check exif image data
    If image orientation is exist, correct orientation of image

    Parameters:
        file_path_name {string} -- A path to the image file

    Returns:
        image {Object} -- An image after correct orientation
    """
    try:
        image = Image.open(file_path_name)
        tags = {}
        # Get exif data of image using exifread library
        with open(file_path_name, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        if "Image Orientation" in tags.keys():
            orientation = tags["Image Orientation"]
            val = orientation.values
            # Image has been rotated 90 degree left and flip to bottom
            if 5 in val:
                val += [4, 8]
            # Image has been rotated 90 degree right and flip to bottom
            if 7 in val:
                val += [4, 6]
            # Image has been rotated 180 degree
            if 3 in val:
                image = image.transpose(Image.ROTATE_180)
            # Image has been flip to bottom
            if 4 in val:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # Image has been rotated 90 degree right
            if 6 in val:
                image = image.transpose(Image.ROTATE_270)
            # Image has been rotated 90 degree left
            if 8 in val:
                image = image.transpose(Image.ROTATE_90)

        return image
    except FileNotFoundError:
        raise FileNotFoundError("Please check your image's path")


def monochromize_image(image, brightness=None):
    """
    Change image to monochrome version with correct brightness
    If brightness is none, use default convert to black & white image
    Else, convert pixel with exact threshold of image

    Parameters:
        image {Object} -- An image would be opened

    Keyword Parameters:
        brightness {float} -- A float value from 0.0 to 1.0 that represents
                              the brightness used as a threshold to divide
                              the grayscale version of the original image
                              into two portions (default: {None})

    Returns:
        image {Object} -- An image have been converted to black & white version
    """
    # Convert list pixels to numpy array, optimize speed
    pixels = np.array(image, dtype=np.uint8)

    # Copy a numpy array, avoid original conflicts
    pixels = pixels.copy()

    if brightness != None:
        if not brightness >= 0 and brightness <= 1:
            raise ValueError("Invalid brightness!")

        threshold = brightness * 255

        # If pixel's value is less than or equal to threshold, convert pixel to black
        pixels[pixels <= threshold] = 0
        # If pixel's value is greater than threshold, convert pixel to white
        pixels[pixels > threshold] = 255

        image = Image.fromarray(pixels)

    # Convert to black & white version
    return image.convert('1', dither=False)


def calculate_brightness(image):
    """
    Calculate brightness of greyscale image

    Parameters:
        image {Object} -- An image would be opened

    Returns:
        brightness {float} -- A float number from 0.0 to 1.0,
                                  display image darker or lighter
    """
    # Get histogram of greyscale image
    histogram = image.convert('L').histogram()
    # Sum all pixel's value
    all_pixels_value = sum(histogram)
    # Prepare default brightness and histogram's scale
    brightness = histogram_scale = len(histogram)

    for pixel_index in range(0, histogram_scale):
        # Get ratio (not percent) of one pixel's value compare with all of pixel's value
        ratio = histogram[pixel_index] / all_pixels_value
        # Get focus area of pixel by subtract focusing ratio of pixel's value
        # Based on distance of one pixel's value with all of another pixel's value
        brightness -= ratio * (histogram_scale - pixel_index)

    return 1 if brightness == 255 else brightness / histogram_scale


def filter_visible_sprites(sprites, min_surface_area):
    """
    Filter visible sprites

    Parameters:
        sprites {dict} -- A list of Sprite object
        min_surface_area {int} -- An integer representing the minimal surface
                                  area of a sprite's bounding box to consider
                                  this sprite as visible.

    Returns:
        visible_sprites {dict} -- A dictionary of visible sprites
    """
    visible_sprites = {}
    for label, sprite in sprites.items():
        # Get all sprite have surface greater than minimum surface
        if sprite.surface >= min_surface_area:
            visible_sprites[label] = sprite

    return visible_sprites


def filter_square_sprites(sprites, similarity_threshold):
    """
    Filter square sprites

    Parameters:
        sprites {dict} -- A list of Sprite object
        similarity_threshold {float} -- A float number between 0.0 and 1.0 of
                                        the relative difference of the width and height
                                        of the sprite's boundary box over which
                                        the sprite is not considered as a square

    Returns:
        square_sprites {dict} -- A dictionary of square sprites
    """
    # Raise invalid similarity threshold
    if not 1 >= similarity_threshold >= 0:
        raise ValueError("Invalid similarity threshold")

    square_sprites = {}
    for label, sprite in sprites.items():
        # Get relative difference of square sprite (compare width & height)
        relative_difference_square = get_relative_difference(
            sprite.width, sprite.height)
        # Get all sprite have relative difference less than threshold
        if relative_difference_square <= similarity_threshold:
            square_sprites[label] = sprite

    return square_sprites


def filter_dense_sprites(sprites, density_threshold):
    """
    Filter dense sprites

    Arguments:
        sprites {list} -- A list of Sprite object
        density_threshold {float} -- A float number between 0.0 and 1.0 representing the
                                     percentage difference between the number of pixels
                                     of a sprite and the surface area of the boundary box
                                     of this sprite, over which the sprite is considered as dense.

    Returns:
        dense_sprites {dict} -- A dictionary of dense sprites
    """
    # Raise invalid density threshold
    if not 1 >= density_threshold >= 0:
        raise ValueError("Invalid density threshold")

    dense_sprites = {}
    for label, sprite in sprites.items():
        # Get all sprite have density greater than density threshold
        if sprite.density >= density_threshold:
            dense_sprites[label] = sprite

    return dense_sprites


def group_sprites_by_similar_size(sprites, similar_size_threshold):
    """
    Group sprites with similar size, compare with threshold

    Parameters:
        sprites {list} -- A list of Sprite object
        similar_size_threshold {float} -- A float number between 0.0 and 1.0
                                            representing the relative difference
                                            between the sizes (surface areas) of
                                            two sprites below which these sprites
                                            are considered similar.

    Returns:
        final_group {list} -- List group of each sprite with similar size
    """
    # Final group of each sprite with similar size
    final_group = []

    # Check sprite traversed to not duplicate itself
    sprite_traversed = []

    for i, (first_key, first_value) in enumerate(sprites.items()):
        if first_value in sprite_traversed:
            continue

        # Create new similar size list for each group
        similar_size_group = []
        has_similar_size = False
        for j, (second_key, second_value) in enumerate(sprites.items()):
            if j <= i:
                continue
            if second_value in sprite_traversed:
                continue

            # Get relative difference size (surface)
            first_surface_area = first_value.surface
            second_surface_area = second_value.surface
            relative_difference_size = get_relative_difference(
                first_surface_area, second_surface_area)

            # Get all sprite have relative difference size less than threshold
            if relative_difference_size <= similar_size_threshold:
                has_similar_size = True
                similar_size_group.append(second_value)
                sprite_traversed.append(second_value)

        # Check the current sprite has similar size with another sprite
        if has_similar_size:
            similar_size_group.append(first_value)
            sprite_traversed.append(first_value)

            # Make sure three sprites can create two tuples
            # Remove list contains only two sprites
            if len(similar_size_group) >= 3:
                final_group.append(similar_size_group)

    return final_group


def group_sprites_by_similar_distance(similar_size_group, similar_distance_threshold):
    """
    Group sprite with similar distance, after group srite with similar size

    Parameters:
        similar_size_group {list} -- A big group contain nested group,
                                      each sprite in nested group has similar size
        similar_distance_threshold {float} -- A float number between 0.0 and 1.0
                                              representing the relative difference
                                              between the distances from the sprites of
                                              2 pairs below which these pairs are
                                              considered having similar distance.

    Returns:
        final_group {list} -- A big group contain nested group,
                              each tuple in nested group has similar distance
    """
    # Convert similar size group from list to numpy array, optimize speed
    similar_size_group = np.array(similar_size_group)

    # Final group of each sprite with similar size
    final_group = []

    for index in range(len(similar_size_group)):
        # Get all distance of each pairs of sprite
        pairs_distance_dict = {}
        pairs_distance_dict = get_group_sprite_pairs_distance(
            similar_size_group[index])

        # Check tuple sprite traversed to not duplicate itself
        tuple_sprite_traversed = []
        for i, (first_key, first_value) in enumerate(pairs_distance_dict.items()):
            if first_key in tuple_sprite_traversed:
                continue

            # Create similar distance list for each group
            similar_distance_group = []
            has_similar_distance = False
            for j, (second_key, second_value) in enumerate(pairs_distance_dict.items()):
                if j <= i:
                    continue
                if second_key in tuple_sprite_traversed:
                    continue

                relative_difference_distance = get_relative_difference(
                    first_value, second_value)

                # Get all sprite have relative difference distance less than threshold
                if relative_difference_distance <= similar_distance_threshold:
                    has_similar_distance = True
                    similar_distance_group.append(second_key)
                    tuple_sprite_traversed.append(second_key)

            # Check the current sprite has similar distance with another sprite
            if has_similar_distance:
                similar_distance_group.append(first_key)
                tuple_sprite_traversed.append(first_key)
                final_group.append(similar_distance_group)

    return final_group


def get_group_sprite_pairs_distance(group):
    """
    Get group each pairs of sprite and calculate distance for itself

    Arguments:
        group {list} -- A list of sprite with similar size

    Returns:
        pairs_distance_dict {dict} -- A dictionary with pairs of sprite and distance of itself
    """
    pairs_distance_dict = {}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            # Calculate distance of current sprite with the next sprite in list
            pairs_distance_dict[(group[i], group[j])] = calculate_distance(
                group[i].centroid, group[j].centroid)

    return pairs_distance_dict


def group_sprites_by_similar_size_and_distance(sprites, similar_size_threshold, similar_distance_threshold):
    """
    Group sprite by similar size and distance

    Arguments:
        sprites {list} -- A list of Sprite object
        similar_size_threshold {float} -- A float number between 0.0 and 1.0
                                            representing the relative difference
                                            between the sizes (surface areas) of
                                            two sprites below which these sprites
                                            are considered similar.
        similar_distance_threshold {float} -- A float number between 0.0 and 1.0
                                              representing the relative difference
                                              between the distances from the sprites of
                                              2 pairs below which these pairs are
                                              considered having similar distance.

    Returns:
        [type] -- [description]
    """
    # Raise invalid similar size and similar distance threshold
    if not 1 >= similar_size_threshold >= 0:
        raise ValueError("Invalid similar size threshold!")
    if not 1 >= similar_distance_threshold >= 0:
        raise ValueError("Invalid similar distance threshold!")

    # Get similar size group first
    similar_size_group = group_sprites_by_similar_size(
        sprites, similar_size_threshold)
    # Get similar distance group after has similar size group
    similar_size_distance_group = group_sprites_by_similar_distance(
        similar_size_group, similar_distance_threshold)

    return similar_size_distance_group


def calculate_distance(first_point, second_point):
    """
    Get distance of two points

    Arguments:
        first_point {tuple} -- First point to calculate distance
        second_point {tuple} -- Second point to calculate distance

    Returns:
        distance {float} -- A float number representing for distance of two points
    """
    return distance.euclidean(first_point, second_point)


def get_percentage_difference(first_value, second_value):
    """
    Calculate percentage difference of two values

    Parameters:
        first_value {number} -- First number to calculate percentage difference
        second_value {number} -- Second number to calculate percentage difference

    Returns:
        percentage_difference {float} -- A float number from 0.0 to 1.0
                                         representing for percentage difference
    """
    return second_value / first_value if first_value > second_value else first_value / second_value


def get_relative_difference(first_value, second_value):
    """
    Calculate relative difference of two values

    Arguments:
        first_value {number} -- First number to calculate relative difference
        second_value {number} -- Second number to calculate relative difference

    Returns:
        relative_difference {float} -- A float number from 0.0 to 1.0
                                       representing for relative difference
    """
    return abs(second_value - first_value) / max((first_value, second_value))


def search_position_detection_patterns(sprite_pairs, orthogonality_threshold):
    """
    Get list of patterns create to be an orthogonal angle

    Arguments:
        sprite_pairs {list} -- A list of pairs of sprites
        orthogonality_threshold {float} -- A float number between 0.0 and 1.0 of
                                            the relative difference between the angle
                                            of two pairs of sprites less or equal which
                                            the two pairs are considered orthogonal

    Returns:
        final_list {list} -- A list of tuple three patterns create an orthogonal angle
    """
    # Setup const number of orthogonal degree
    ORTHOGONAL_DEGREE = 90

    final_list = []
    for index in range(len(sprite_pairs)):
        sprite_checking = sprite_pairs[index]
        for i in range(len(sprite_checking)):
            for j in range(i + 1, len(sprite_checking)):
                # Get common point of two pairs of sprite
                common_point = list(
                    set(sprite_checking[i]).intersection(set(sprite_checking[j])))
                # If none of common point, next
                if len(common_point) == 0:
                    continue

                # Common point is upper left sprite pattern of QR code
                upper_left_sprite = common_point[0]

                # Get upper right sprite pattern and lower left sprite pattern
                upper_right_sprite, lower_left_sprite = detect_upper_right_and_lower_left(
                    sprite_checking, i, j)

                # Calculate angle of three sprites
                angle = calculate_angle(
                    upper_left_sprite.centroid, upper_right_sprite.centroid, lower_left_sprite.centroid)
                relative_difference_orthogonal = get_relative_difference(
                    angle, ORTHOGONAL_DEGREE)

                # Get all tuple have relative difference orthogonal less than threshold
                if relative_difference_orthogonal <= orthogonality_threshold:
                    final_list.append(
                        (upper_left_sprite, upper_right_sprite, lower_left_sprite))

    return final_list


def detect_upper_right_and_lower_left(sprite_checking, i, j):
    """
    Detect upper right sprite and lower left sprite

    Arguments:
        sprite_checking {list} -- A list of Sprite object are currently checking
        i {number} -- Index of first sprite
        j {number} -- Index of second sprite

    Returns:
        upper_right_sprite, lower_left_sprite -- Last part of QR code's pattern
    """
    # If symmetric of left point's label greater than right point's label
    # Upper right is symmetric left, lower left is symmetric right
    # Else opposite
    if list(set(sprite_checking[i]) - set(sprite_checking[j]))[0].label \
            > list(set(sprite_checking[j]) - set(sprite_checking[i]))[0].label:
        upper_right_sprite = list(
            set(sprite_checking[i]) - set(sprite_checking[j]))[0]
        lower_left_sprite = list(
            set(sprite_checking[j]) - set(sprite_checking[i]))[0]
    else:
        upper_right_sprite = list(
            set(sprite_checking[j]) - set(sprite_checking[i]))[0]
        lower_left_sprite = list(
            set(sprite_checking[i]) - set(sprite_checking[j]))[0]

    return upper_right_sprite, lower_left_sprite


def calculate_angle(common_point, first_point, second_point):
    """
    Calculate angle of three points (create two vectors, with one common point)

    Arguments:
        common_point {tuple} -- Origin point of two vectors
        first_point {tuple} -- First point left of two vectors
        second_point {tuple} -- Second point left of two vectors

    Returns:
        angle {number} -- An angle with degree units
    """
    # Calculate two vectors
    first_vector = (first_point[0] - common_point[0],
                    first_point[1] - common_point[1])
    second_vector = (second_point[0] - common_point[0],
                     second_point[1] - common_point[1])

    # Scalar of two vectors
    scalar = first_vector[0] * second_vector[0] + \
        first_vector[1] * second_vector[1]

    # Calculate length of vector
    first_vector_length = calculate_length_of_vector(first_vector)
    second_vector_length = calculate_length_of_vector(second_vector)

    # Volumetric length of two vectors
    volumetric_length = first_vector_length * second_vector_length

    # Return degree of angle
    return math.degrees(math.acos(scalar / volumetric_length))


def calculate_length_of_vector(vector):
    """
    Calculate length of vector

    Arguments:
        vector {tuple} -- A vector

    Returns:
        length_of_vector {float} -- A float number representing the length of vector
    """
    return math.sqrt(vector[0]**2 + vector[1]**2)


def filter_matching_inner_outer_finder_patterns(finder_patterns):
    """
    Filter inner and outer patterns

    Arguments:
        finder_patterns {list} -- List of three patterns of QR code

    Returns:
        outer_patterns {tuple} -- Three outer patterns of QR code
    """
    for i in range(len(finder_patterns)):
        for j in range(i + 1, len(finder_patterns)):
            if finder_patterns[i][0].centroid[0] != finder_patterns[j][0].centroid[0]:
                continue
            # Calculate
            first_surface_area = finder_patterns[i][0].surface
            second_surface_area = finder_patterns[j][0].surface

            # Return outer patterns of QR code
            return finder_patterns[i] if first_surface_area > second_surface_area else finder_patterns[j]


def crop_qr_code_image(image, outer_patterns, label_map):
    """
    Rotate and crop QR code image

    Arguments:
        image {Object} -- Image of QR code to rotate and crop
        outer_patterns {tuple} -- Three patterns of QR code

    Returns:
        image_cropped {Object} -- An image after rotate and crop
    """
    # Upper left pattern of QR code
    upper_left_sprite = outer_patterns[0]
    # Upper right pattern of QR code
    upper_right_sprite = outer_patterns[1]
    # Lower left pattern of QR code
    lower_left_sprite = outer_patterns[2]

    # Get angle to rotate
    angle = detect_angle_to_rotate_qr_code(
        upper_left_sprite, lower_left_sprite)

    # Choose center point to rotate
    center_point = upper_left_sprite.centroid

    # Rotate image
    image_rotated = image.rotate(angle, center=center_point)

    # Get new coordinates
    new_upper_left_centroid = get_new_coords_after_rotate(
        upper_left_sprite.centroid, center_point, angle)

    new_upper_right_centroid = get_new_coords_after_rotate(
        upper_right_sprite.centroid, center_point, angle)

    new_lower_left_centroid = get_new_coords_after_rotate(
        lower_left_sprite.centroid, center_point, angle)

    # Set up box's position
    left = new_upper_left_centroid[0]
    upper = new_upper_left_centroid[1]
    right = new_upper_right_centroid[0]
    lower = new_lower_left_centroid[1]

    # Get box with parameter
    box = (left - upper_left_sprite.width // 2, upper - upper_left_sprite.width //
           2, right + upper_right_sprite.width // 2, lower + lower_left_sprite.width // 2)

    # Crop image
    image_cropped = image_rotated.crop(box)

    return image_cropped


def get_new_coords_after_rotate(old_point, center_point, angle):
    """
    Return new coordinates of a point P(x, y) after 
    rotate an angle alpha to be a point P'(x', y')

    Arguments:
        old_point {tuple} -- A point with coordinates before rotate
        center_point {tuple} -- A point was chosen to be an origin point
        angle {float} -- An angle to rotate

    Returns:
        coordinates {tuple} -- New coordinates of point P(x, y) -> P'(x', y')
    """
    coordinates = ()
    # Convert degree to radian
    angle = -angle
    angle = angle * (math.pi / 180)
    # Diaphragm degrees
    new_x = (old_point[0] - center_point[0]) * round(math.cos(angle), 3) - \
        (old_point[1] - center_point[1]) * \
        round(math.sin(angle), 3) + center_point[0]
    # Bounce degrees
    new_y = (old_point[0] - center_point[0]) * round(math.sin(angle), 3) + \
        (old_point[1] - center_point[1]) * \
        round(math.cos(angle), 3) + center_point[1]
    # New coordinates
    coordinates = (round(new_x), round(new_y))
    
    return coordinates


def detect_angle_to_rotate_qr_code(upper_left_sprite, lower_left_sprite):
    """
    Detect an angle to rotate QR code compare with vertical axis in a right way

    Arguments:
        upper_left_sprite {Object} -- Upper left pattern of QR code
        lower_left_sprite {Object} -- Lower left pattern of QR code

    Keyword Arguments:
        trade_off {float} -- Number to reduce errors when calculating

    Returns:
        angle {int} -- An angle to rotate QR code in a right way
    """
    angle = None
    # QR Code in the right way
    if upper_left_sprite.centroid[0] == lower_left_sprite.centroid[0] \
            and upper_left_sprite.centroid[1] > lower_left_sprite.centroid[1]:
        angle = 0
    # QR Code was rotated to the right 180 degrees, need to rotate left back
    elif upper_left_sprite.centroid[0] == lower_left_sprite.centroid[0] \
            and upper_left_sprite.centroid[1] < lower_left_sprite.centroid[1]:
        angle = 180
    # QR Code was rotated to the right 270 degrees, need to rotate left back
    elif upper_left_sprite.centroid[1] == lower_left_sprite.centroid[1] \
            and upper_left_sprite.centroid[0] > lower_left_sprite.centroid[0]:
        angle = 270
    # QR Code was rotated to the right 90 degrees, need to rotate left back
    elif upper_left_sprite.centroid[1] == lower_left_sprite.centroid[1] \
            and upper_left_sprite.centroid[0] < lower_left_sprite.centroid[0]:
        angle = 90
    # In general case
    else:
        # Detect common point to rotate
        if upper_left_sprite.centroid[1] < lower_left_sprite.centroid[1]:
            common_point = upper_left_sprite.centroid
            first_point = (
                upper_left_sprite.centroid[0], lower_left_sprite.centroid[1])
            second_point = lower_left_sprite.centroid
            angle = calculate_angle(common_point, first_point, second_point)
        elif upper_left_sprite.centroid[1] > lower_left_sprite.centroid[1]:
            common_point = lower_left_sprite.centroid
            first_point = (
                lower_left_sprite.centroid[0], upper_left_sprite.centroid[1])
            second_point = upper_left_sprite.centroid
            angle = 180 - \
                calculate_angle(common_point, first_point, second_point)

    return angle
