# QR Code Reader

![Hitman Agent 47](./resources/hitman_agent_47.png)

## What is a QR Code?

A [Quick Response (QR) code](https://en.wikipedia.org/wiki/QR_code) is two-dimensional [**barcode**](https://en.wikipedia.org/wiki/Barcode) first [designed in 1994](https://www.youtube.com/watch?v=LS1wrEv-fjk) for the automotive industry in Japan. A **barcode** is a machine-readable optical label that represents data.

A QR code consists of black squares arranged in a square grid on a white background, which can be read by any devices that have an embedded camera (QR code **scanner**), typically a smartphone. Data are then extracted from patterns that are present in both horizontal and vertical components of the image.

One common usage of a QR code is to open a website in a web browser on the device that captures the QR code:

![QR Code Intek Institute Website URI](./resources/qr_code_intek_edu_vn_logo.png)

The amount of data that can be stored in a QR code symbol depends on the **type** of the data (numeric, alphanumeric, binary, etc.), the **version** of the QR code, and the **error correction** level:

| Data type    | Max. characters |
| ------------ | --------------- |
| Numeric      | 7,089           |
| Alphanumeric | 4,296           |
| Binary       | 2,953           |

A QR code uses the [Reed–Solomon error correction](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) algorithm with four error correction levels. The higher the error correction level, the less storage capacity. The following table lists the approximate error correction capability at each of the four levels:

| Level        | Restorable characters |
| ------------ | --------------------- |
| **L**ow      | 7%                    |
| **M**edium   | 15%                   |
| **Q**uartile | 25%                   |
| **H**igh     | 30%                   |

Examples of QR code that represent the same data but with increasing error correction levels:

| Level `L`                             | Level `M`                             | Level `Q`                             | Level `H`                             |
| ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![](./resources/qr_code_intek_edu_vn_level_l.png) | ![](./resources/qr_code_intek_edu_vn_level_m.png) | ![](./resources/qr_code_intek_edu_vn_level_q.png) | ![](./resources/qr_code_intek_edu_vn_level_h.png) |

Examples of damaged or embellished QR codes that will still scan correctly thanks to error correction:

| Damaged QR Code                               | QR Code with Logo                   |
| --------------------------------------------- | ----------------------------------- |
| ![](./resources/qr_code_intek_edu_vn_level_q_damaged.png) | ![](./resources/qr_code_intek_edu_vn_fancy.png) |

## Your Mission

There are few existing Python implementations of QR code reader ([`pyzbar`](https://github.com/NaturalHistoryMuseum/pyzbar), [`qrtools`](https://github.com/primetang/qrtools), others), but they all depend on other libraries ([OpenCV](https://opencv.org/), [Zbar](http://zbar.sourceforge.net/)), which your grandmother won't be able to install the same way on Windows, Mac, and Linux, in just one click, for sure. And you should consider most of the end users like your grandmother.

Your mission, should you choose to accept it, is to write a QR code finder and decoder in **pure Python**, without the help of any computer vision libraries. A **pure Python** library is a library that only contains Python code, and doesn't include, say, C extensions or code in other languages.

_Note: This **pure Python** implementation of a QR code finder and decoder is not intended to be used for real-time QR code decoding; it could take up to several seconds per image to detect and decode a QR code._

# Photo Orientation and Brightness

The first step consists in finding a form that looks like a QR code in an image, whatever its size, its orientation, or even its color.

For examples:

| Input image                 | Output (monochrome)                  |
| --------------------------- | ------------------------------------ |
| ![](./resources/qr_code_art_island.jpg) | ![](./resources/qr_code_art_island_detected.png) |
| ![](./resources/qr_code_angle.jpg)      | ![](./resources/qr_code_angle_detected.png)      |

You will write the code of these first waypoints in a Python file named `detector.py`.

## Waypoint 1: Load Image and Correct Orientation

When you load the image file of a photo and you save a new version of this image using the Python Imaging Library (PIL), none of the Exif information of the initial image is saved.

The [Exchangeable image file format (Exif)](https://en.wikipedia.org/wiki/Exif) specification defines an [`Orientation` tag](https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html) to indicate the [orientation of the camera](https://www.daveperrett.com/articles/2012/07/28/exif-orientation-handling-is-a-ghetto/) relative to the captured scene. This can be used by the camera either to indicate the orientation automatically by an orientation sensor, or to allow the user to indicate the orientation manually by a menu switch, without actually transforming the image data itself.

Write a function `load_image_and_correct_orientation` that takes an argument `file_path_name` that references an image file of a photo, and that returns an object [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) of the photo which orientation may have been corrected.

For example:

```bash
$ exiftool DSC07314.jpg | grep -i orientation
Orientation                     : Rotate 270 CW
Camera Orientation              : Rotate 270 CW
```

```python
>>> from PIL import Image
>>> image = Image.open('DSC07314.jpg')
>>> image.save('DSC07314_orientation_not_corrected.jpg')
>>> image = load_image_and_correct_orientation('DSC07314.jpg')
>>> image.save('DSC07314_orientation_corrected.jpg')
```

| Initial Image     | Orientation Not Corrected                   | Orientation Corrected                   |
| ----------------- | ------------------------------------------- | --------------------------------------- |
| ![](./resources/DSC07314.jpg) | ![](./resources/DSC07314_orientation_not_corrected.jpg) | ![](./resources/DSC07314_orientation_corrected.jpg) |

_Note: Your Web browser may not correctly display the initial image, because it may not read the Exif information of the photo, and therefore it doesn't automatically display the image with the right orientation. You can click and save the initial image on your computer and display it with a photo application. You will see that the photo is perfectly oriented._

We suggest you to use the 3rd party library [`exifread`](https://github.com/ianare/exif-py) to extract Exif metadata from tiff and jpeg files.

## Waypoint 2: Convert Image to Monochrome

Write a function `monochromize_image` that takes an argument `image` (an object [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)) and that returns one other object `Image` corresponding to the monochrome version of the image passed to this function.

| Color Version                             | [Grayscale](https://en.wikipedia.org/wiki/Grayscale) Version (`L`) | Black & White Version (`P`)                        | Monochromized Version                         |
| ----------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------------------- |
| ![](./resources/qr_code_waypoint2_normal.jpg)         | ![](./resources/qr_code_waypoint2_grayscale.png)                               | ![](./resources/qr_code_waypoint2_black_and_white.png)         | ![](./resources/qr_code_waypoint2_monochrome.png)         |
| ![](./resources/qr_code_waypoint2_normal_details.png) | ![](./resources/qr_code_waypoint2_grayscale_details.png)                       | ![](./resources/qr_code_waypoint2_black_and_white_details.png) | ![](./resources/qr_code_waypoint2_monochrome_details.png) |

_Note: Converting the original image to a [1-bit pixels Black & White](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes) version won't be of a great help. Take a closer look at the [image details](./resources/qr_code_waypoint2_black_and_white_details.png) of the 1-bit pixels black & white version. The black areas are actually covered with a lot of white pixels, while we need to [unify black area](./resources/qr_code_waypoint2_monochrome_details.png) with only black pixels like the monochrome version._

For example:

```python
>>> color_image = Image.open('./photos/DSC07310.jpg')
>>> monochrome_image = monochromize_image(image)
```

Your function `monochromize_image` also accepts an optional argument `brightness`, a float value from `0.0` to `1.0` that represents th brightness used as a [**threshold**](<https://en.wikipedia.org/wiki/Thresholding_(image_processing)>) to divide the grayscale version of the original image into two portions:

- Pixel values (grayscale colors) that are **less than or equal** to this threshold are converted to **black**;
- Pixel values (grayscale colors) **greater than** the threshold are converted to **white**.

_Note: The **threshold** corresponds to `255 * brightness`._

This allows to handle low and high [contrast](<https://en.wikipedia.org/wiki/Contrast_(vision)>) photos for which this threshold should be adapted so that the QR code is clearly visible on the monochrome version.

Example of a high contrast photo:

|                                                                    |                                                                    |                                                                    |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| ![](./resources/qr_code_waypoint2_dark.jpg) Original                           | ![](./resources/qr_code_waypoint2_dark_monochrome_16.png) (`threshold = 16`)   | ![](./resources/qr_code_waypoint2_dark_monochrome_48.png) (`threshold = 48`)   |
| ![](./resources/qr_code_waypoint2_dark_monochrome_80.png) (`threshold = 80`)   | ![](./resources/qr_code_waypoint2_dark_monochrome_112.png) (`threshold = 112`) | ![](./resources/qr_code_waypoint2_dark_monochrome_144.png) (`threshold = 144`) |
| ![](./resources/qr_code_waypoint2_dark_monochrome_176.png) (`threshold = 176`) | ![](./resources/qr_code_waypoint2_dark_monochrome_208.png) (`threshold = 208`) | ![](./resources/qr_code_waypoint2_dark_monochrome_240.png) (`threshold = 240`) |

Example of a low contrast photo:

|                                                                     |                                                                     |                                                                     |
| ------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![](./resources/qr_code_waypoint2_light.jpg) Original                           | ![](./resources/qr_code_waypoint2_light_monochrome_16.png) (`threshold = 16`)   | ![](./resources/qr_code_waypoint2_light_monochrome_48.png) (`threshold = 48`)   |
| ![](./resources/qr_code_waypoint2_light_monochrome_80.png) (`threshold = 80`)   | ![](./resources/qr_code_waypoint2_light_monochrome_112.png) (`threshold = 112`) | ![](./resources/qr_code_waypoint2_light_monochrome_144.png) (`threshold = 144`) |
| ![](./resources/qr_code_waypoint2_light_monochrome_176.png) (`threshold = 176`) | ![](./resources/qr_code_waypoint2_light_monochrome_208.png) (`threshold = 208`) | ![](./resources/qr_code_waypoint2_light_monochrome_240.png) (`threshold = 240`) |

## Waypoint 3: Determine Image Brightness

Depending on the conditions the photo has been captured, whether indoor or outdoor, whether the weather was sunny or cloudy, the brightness of this photo could be totally different. We need to determine, for each particular photo, the appropriate [**brightness**](https://www.allaboutcircuits.com/technical-articles/image-histogram-characteristics-machine-learning-image-processing/) value that our function `monochromize_image` needs to be used to generate a monochrome image with the QR code clearly visible.

The brightness of a photo could be determined from the [**histogram**](https://en.wikipedia.org/wiki/Image_histogram) of the grayscale version of the photo. The histogram corresponds to a list of pixel counts, one for each pixel value in the source image, i.e., the list of pixel counts for each grayscale. The brightness of the image is determined from the number of pixels per gray levels (spatial distribution). If the histogram values are concentrated toward the first levels, the image is darker. If they are concentrated toward the last levels, the image is lighter.

For examples:

|           | Dark                                      | Light                                      |
| --------- | ----------------------------------------- | ------------------------------------------ |
| Photo     | ![](./resources/qr_code_waypoint2_dark.jpg)           | ![](./resources/qr_code_waypoint2_light.jpg)           |
| Histogram | ![](./resources/qr_code_waypoint3_dark_histogram.png) | ![](./resources/qr_code_waypoint3_light_histogram.png) |

<em>Note: The [**contrast**](<https://en.wikipedia.org/wiki/Contrast_(vision)>) is the difference in luminance that makes an object distinguishable.</em>

Write a function `calculate_brightness` that takes an argument `image` (an object [`Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)) and that returns a float ranging from `0.0` to `1.0`.

For example:

```python
>>> calculate_brightness(Image.open('./samples/20190820_113850.jpg'))
0.40121496996211214
>>> calculate_brightness(Image.open('./samples/20190820_114009.jpg'))
0.2647252011140948
```

_Note: Your function may return slightly different brightness values than these examples. That is totally fine as long as the brightness values returned by your function `calculate_brightness` allows your function `monochromize_image` to generate a monochrome version of your photos with the QR code clearly visible._

# QR Code Detection

A QR Code consists of an array of square modules arranged in an overall square pattern, including a unique **finder pattern** composed of three identical components (also known as **position detection patterns**) located at the left upper, right upper, and left lower corners of the symbol and intended to assist in easy location of its position, size and inclination.

![QR Code Symbol Structure](./resources/qr_code_symbol_structure.png)

To detect whether an image contains one or more QR codes, we need to find one or more **finder patterns** in this image: a list of combinations of 3 **position detection patterns** each.

## Waypoint 4: Filter Visible Sprites

The sprites corresponding to **position detection patterns** need to be clearly visible in the image, meaning their surface area need to be big enough. If they were too small, we would have difficulties to decode the QR code by reading its _hieroglyphs_ (smaller dots).

Write a function `filter_visible_sprites` that takes two arguments:

- `sprites`: A list of sprites.

- `min_surface_area`: An integer representing the minimal surface area of a sprite's bounding box to consider this sprite as visible.

The function `filter_visible_sprites` returns a list of the `Sprite` objects which surface area is equal to or larger than the specified minimal surface. We will use the function `filter_visible_sprites` to only retain sprites that have a surface visible to the naked eye.

_Note: To drastically improve the performance of the calculation, the surface of a sprite is approximate to the area of its bounding box._

Let's take an example with the following photo. We found **`20,040` sprites** in this photo:

| Photo                    | Sprites                          |
| ------------------------ | -------------------------------- |
| ![](./resources/20190910_162611.jpg) | ![](./resources/20190910_162611_sprites.png) |

We apply a filter to the list to only retain sprites with a surface area equal to or greater than `256` pixels:

```python
>>> visible_sprites = filter_visible_sprites(sprites, 256)
>>> print(len(visible_sprites))
3781
```

![](./resources/20190910_162611_visible_sprites_256.png)

If the minimal surface area threshold (`min_surface_area`) is too high, your function may not return the sprites corresponding to the **position detection patterns** of a QR code, and therefore your library won't be able to detect the QR code in the image.

For example:

```python
>>> visible_sprites = filter_visible_sprites(sprites, 16384)
>>> print(len(visible_sprites))
118
```

![](./resources/20190910_162611_visible_sprites_16384.png)

If the minimal surface area threshold (`min_surface_area`) is too low, your function will return way to many sprites which processing may slow down the detection of QR codes in the image.

For example:

```python
>>> visible_sprites = filter_visible_sprites(sprites, 64)
>>> print(len(visible_sprites))
5764
```

![](./resources/20190910_162611_visible_sprites_64.png)

_Note: You **MAY** get different results depending on the brightness threshold you use to convert the original photo to a monochrome image._

How could we determine the value of the minimal surface area that could be visible to the naked eye? It **SHOULD** depend on the resolution of the size of the image.

## Waypoint 5: Filter Square Sprites

The **position detection patterns** of a QR code have a square shape. We need to only retain sprites who have a square shape.

Write a function `filter_square_sprites` that takes two arguments:

- `sprites`: A list of `Sprite` objects.

- `similarity_threshold`: A float number between `0.0` and `1.0` of the relative difference of the width and height of the sprite's boundary box over which the sprite is not considered as a square.

The function `filter_square_sprites` returns a list of the `Sprite` objects which boundary box is almost a square.

![](./resources/20190910_162611_square_sprites.png)

_Note: To drastically improve the performance of the calculation, the shape of a sprite is approximate to the shape of its bounding box._

## Waypoint 6: Filter Dense Sprites

The previous waypoints suppose that the size and the shape of a sprite is similar to the size and the shape of its bounding box. This assumption was used to improve the performance of calculation. However, the large the bounding box of sprites, the less true this assumption.

For example, the following sprite has a square bounding box. However, the sprite itself is not a square shape:

![Sprite with Square Bounding Box](./resources/sprite_with_square_bounding_box.png)

We need to consider how dense is a sprite, meaning the ratio between the number of pixels that compose a sprite and the surface area of the boundary box of this sprite. We will only retain sprites that are dense.

Write a function `filter_dense_sprites` that takes two arguments:

- `sprites`: A list of `Sprite` objects.

- `density_threshold`: A float number between `0.0` and `1.0` representing the relative difference between the number of pixels of a sprite and the surface area of the boundary box of this sprite, over which the sprite is considered as dense.

The function `filter_dense_sprites` returns a list of the `Sprite` objects that are dense.

![](./resources/20190910_162611_dense_sprites.png)

## Waypoint 7: Group Sprites by Similar Size and Distance

The sprites representing the 3 **position detection patterns** of a QR code have similar size and one of these 3 sprites is equidistant from the 2 others. Among the 3 possible pairs of these sprites, 2 ot them have the same distance.

We need to group pairs of sprites with similar size and with similar distance. For example, let's consider the following sprites:

![Pairs of Sprites with Similar Size and Equidistant](./resources/similar_size_and_equidistant_sprite_pairs.jpg)

We need to group these pairs of sprites (with no particular order) as follows:

```python
[
    [(1, 2), (2, 7), (3, 5), (7, 11)],
    [(4, 6), (8, 9), (9, 10)]
]
```

- **Group 1**: The sprites `1`, `2`, `3`, `5`, `7`, and `11` have similar size. Each pair `(1, 2)`, `(2, 7)`, `(3, 5)` and `(7, 11)` has similar distance between its two sprites.

- **Group 2**: The sprites `4`, `6`, `8`, `9`, and `10` have similar size. Each pair `(4, 6)`, `(8, 9)`, and `(9, 10)` has similar distance between its two sprites.

_Note: A group **MUST** contain at least 2 pairs of sprites._

Write a function `group_sprites_by_similar_size_and_distance` that takes three arguments:

- `sprites`: A list of `Sprite` objects.

- `similar_size_threshold`: A float number between `0.0` and `1.0` representing the [**relative difference**](https://en.wikipedia.org/wiki/Relative_change_and_difference) between the sizes (surface areas) of two sprites below which these sprites are considered similar.

- `similar_distance_threshold`: A float number between `0.0` and `1.0` representing the **relative difference** between the distances from the sprites of 2 pairs below which these pairs are considered having similar distance.

The function `group_sprites_by_similar_size_and_distance` returns a list of groups (lists) of pairs of sprites with the following properties:

- the sprites of a same group have similar size.

- the distance between the sprites of each pair of a same group is equivalent.

## Waypoint 8: Search Position Detection Patterns

3 sprites may possibly correspond to a finder pattern of a QR code if they form an angle almost **orthogonal**.

_Note: Indeed these 3 sprites **MUST** be dense, **MUST** have similar size (surface area), and 2 of these sprites **MUST** be equidistant to the 3rd sprite._

![](./resources/sprite_pairs_orthogonality.jpg)

Write a function `search_position_detection_patterns` that takes two arguments:

- `sprite_pairs`: A list of pairs of sprites.

- `orthogonality_threshold`: A float number between `0.0` and `1.0` of the relative difference between the angle of two pairs of sprites less or equal which the two pairs are considered orthogonal.

The function returns a list of 3 sprites (a tuple `upper_left_sprite, upper_right_sprite, lower_left_sprite`) that possibly corresponds to the position detection patterns of a QR code:

- `upper_left_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the upper left corner of the QR code.

- `upper_right_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the upper right corner of the QR code.

- `lower_left_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the lower left corner of the QR code.

_Note: The function expects that the list `sprite_pairs` contains sprites of similar size, and the distance between the two sprites of a pair is similar for each pair; the function doesn't need to verify this assertion._

## Waypoint 9: Filter Matching Inner and Outer Finder Patterns

Write a function `filter_matching_inner_outer_finder_patterns` that takes one argument `finder_patterns` (a list of tuple `upper_left_sprite, upper_right_sprite, lower_left_sprite`). Each tuple corresponds to 3 sprites that may possibly correspond to a finder pattern of a QR code.

The function searches each tuple (**outer finder pattern**) that contains another tuple (**inner finder pattern**).

A tuple contains another tuple if each sprite `upper_left_sprite`, `upper_right_sprite`, and `lower_left_sprite` of this first tuple contains each sprite respectively `upper_left_sprite`, `upper_right_sprite`, and `lower_left_sprite` of the second tuple.

A sprite contains another sprite if the bounding box of this first sprite contains the bounding box of the second sprite.

![](./resources/inner_and_outer_sprites.jpg)

The function `filter_matching_inner_outer_finder_patterns` returns a list of tuples corresponding to the **outer finder patterns**.

![](./resources/20190910_162611_outer_finder_pattern.png)

## Waypoint 10: Crop the QR Code Image Portion

We are almost done.

Write a function `crop_qr_code_image` that takes two arguments `image` and a tuple `upper_left_sprite, upper_right_sprite, lower_left_sprite` where:

- `image`: A `PIL.Image` instance of the photo where the QR code has been found

- `upper_left_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the upper left corner of the QR code.

- `upper_right_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the upper right corner of the QR code.

- `lower_left_sprite`: A `Sprite` object corresponding to the Position Detection Pattern located at the lower left corner of the QR code.

The function returns a `PIL.Image` corresponding to the portion of the QR code rotated and cropped from `image`.

_Hint: We remind you how to calculate the new coordinates `(x', y')` of a point `P(x, y)` that is rotated around a point `O(x, y)` by an angle `θ`:_

![](./resources/point_rotation_formula.png)

| Original Image Rotated                          | QR Code Image Portion Cropped                   |
| ----------------------------------------------- | ----------------------------------------------- |
| ![](./resources/20190910_162611_rotated_finder_pattern.png) | ![](./resources/20190910_162611_cropped_finder_pattern.png) |

# QR Code Decoder

## Waypoint 11: Find QR Code Symbol Version

A QR code symbol is basically a matrix composed of cells with dark or light area, also known as **modules**. A **module** is comparable to a [binary digit (**bit**)](https://en.wikipedia.org/wiki/Bit). A **module** with a dark area corresponds to the **bit** `1`, while **module** with a light area corresponds to the **bit** `0`.

There are 40 sizes of QR code symbol referred to as **Version 1**, **Version 2**, etc. **Version 1** measures 21x21 **modules**, **Version 2** measures 25x25 **modules** and so on increasing in steps of 4 **modules** per side up.

As specified in the _QR Code Standardization ISO 18004_ document:

> _The finder pattern shall consist of three identical Position Detection Patterns located at the upper left, upper right and lower left corners of the symbol respectively. Each Position Detection Pattern may be viewed as three superimposed concentric squares and is constructed of dark 7x7 modules, light 5x5 modules and dark 3x3 modules. The ratio of module widths in each Position Detection Pattern is `1:1:3:1:1`._

![Structure of Position Detection Pattern](./resources/position_detection_pattern_stucture.png)

Write a function `find_qr_code_version` that takes an argument `image` (an object `PIL.Image` of a QR code symbol) and an argument `position_detection_patterns` (a list of objects representing the position detection patterns of the QR code).

The function `find_qr_code_version` returns a tuple of two integers `(version, width)` corresponding respectively to the version of this QR code symbol and the number of modules (on one row of the QR code symbol).

For example:

```python
>>> qr_codes = find_qr_codes('DSC07310.jpg')
>>> qr_code = qr_codes[0]
>>> print(find_qr_code_version(qr_code.image, qr_code.position_detection_patterns))
(2, 25)
```

## Waypoint 12: Convert a QR Code Symbol to an Array of Bits

Write a function `convert_qr_code_to_bit_array` that takes an argument `image` (an object `PIL.Image` of QR code symbol) and an argument `position_detection_patterns` (a list of objects representing the position detection patterns of the QR code).

The function `convert_qr_code_to_bit_array` returns an array of integers `0` and `1` representing the modules of the QR code symbol.

For example:

```python
>>> qr_codes = find_qr_codes('DSC07310.jpg')
>>> qr_code = qr_codes[0]
>>> print(convert_qr_code_to_bit_array(qr_code.image, qr_code.position_detection_patterns))
[
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]
]
```

## Waypoint 13: Decode a QR Code Symbol

Write a function `decode_qr_code_from_array` that takes an argument `qr_code_array` (an array of integers `0` and `1` representing the modules of the QR code symbol) and that returns the value of this QR code symbol.

For example:

```python
>>> qr_codes = find_qr_codes('DSC07310.jpg')
>>> qr_code = qr_codes[0]
>>> qr_code_array = convert_qr_code_to_bit_array(qr_code.image, qr_code.position_detection_patterns)
>>> decode_qr_code_from_array(qr_code_array)
'https://intek.edu.vn'
```

Enjoy!

![Fake QR Code on Cappuccino](./resources/qr_code_cappuccino.jpg)
