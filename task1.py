# OpenCV version 4.5.4
"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np

# import sys
# large_width = 400
# np.set_printoptions(linewidth=large_width)
# np.set_printoptions(threshold=sys.maxsize)


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # Best 24 with grid (4, 6) - 0.958
    # 2nd Best 32 with grid (5, 8) - 0.945
    scaled_size = 24
    grid_size = (4, 6)

    # Template Features
    enrollment(characters, grid_size, scaled_size)

    json_map = detection(test_img, grid_size, scaled_size)

    results = recognition(json_map, characters)

    return results


def enrollment(characters, grid_size, scaled_size):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    template_zone_vectors = []
    for char in characters:
        img: np.ndarray = char[1]

        # Trimming white borders
        del_cols = np.where(np.mean(img, axis=0) > 245)
        img = np.delete(img, del_cols, axis=1)
        del_rows = np.where(np.mean(img, axis=1) > 245)
        img = np.delete(img, del_rows, axis=0)

        # Scaling image to fixed size
        if (img.shape[0] >= img.shape[1]):
            img = cv2.resize(img, (round((scaled_size * img.shape[1]) / img.shape[0]), scaled_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (scaled_size, round((scaled_size * img.shape[0]) / img.shape[1])), interpolation=cv2.INTER_NEAREST)

        # Generating Template Zoning Features
        # Overlays the image onto fixed scaled template
        temp_img = np.ones((scaled_size, scaled_size), dtype=np.uint8) * 255
        x_lower, y_lower = temp_img.shape[0] // 2 - img.shape[0] // 2, temp_img.shape[1] // 2 - img.shape[1] // 2
        x_higher, y_higher = temp_img.shape[0] // 2 + img.shape[0] // 2, temp_img.shape[1] // 2 + img.shape[1] // 2
        if(img.shape[0] % 2 != 0):
            x_lower -= 1
        if (img.shape[1] % 2 != 0):
            y_lower -= 1
        temp_img[x_lower: x_higher, y_lower: y_higher] = img
        img = temp_img
        del temp_img

        # Calculating pixel count for each zone
        zone_vector = []
        zone_shape = (img.shape[0] // grid_size[0], img.shape[1] // grid_size[1])  # eg 9,8
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                img_zone = img[i * zone_shape[0]:(i * zone_shape[0] + zone_shape[0]), j * zone_shape[1]:(j * zone_shape[1] + zone_shape[1])]
                img_zone = np.where(img_zone <= 100, 1, 0)
                zone_vector.append(int(np.sum(img_zone)))
        template_zone_vectors.append(zone_vector)

    zoning_features = {"template_zone_vectors": template_zone_vectors}
    with open(os.path.join('features', 'zoning_features.json'), "w") as file:
        json.dump(zoning_features, file)


def detection(data, grid_size, scaled_size):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    bw_threshold = 100

    image = np.array(data)
    image = np.where(image <= bw_threshold, 0, 255)
    link_pairs = []
    labels = np.zeros((image.shape), dtype=np.uint64)
    label_counter = 1

    # First Pass
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (image[row][col] < 255):
                neighbors = np.zeros((2,), dtype=np.uint64)  # west is 0, north is 1
                if (col - 1 >= 0):
                    if (image[row][col - 1] == 0 and labels[row][col - 1] != 0):  # if neighbor has same value as current and has label
                        neighbors[0] = labels[row][col - 1]  # add west label
                if (row - 1 >= 0):
                    if (image[row - 1][col] == 0 and labels[row - 1][col] != 0):
                        neighbors[1] = labels[row - 1][col]  # add north label

                if (neighbors == np.zeros((2,))).all():  # no neighbors
                    labels[row][col] = label_counter
                    label_counter += 1
                else:
                    if (neighbors == np.zeros((2,))).any():  # any-one neighbor has no label
                        labels[row][col] = np.max(neighbors)
                    else:
                        labels[row][col] = np.min(neighbors)
                        if (neighbors[0] != neighbors[1]):  # both neighbors are not same, but are linked
                            link_pairs.append((neighbors[0], neighbors[1]))

    # Second Pass
    link_map = [i for i in range(label_counter)]
    link_pairs.sort()
    for i, j in link_pairs:
        p = i
        while (link_map[p] != p):
            p = link_map[p]
        c = j
        while (link_map[c] != c):
            c = link_map[c]
        if p != c:
            link_map[c] = p

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (image[row][col] < 255):
                root = labels[row][col]
                while (link_map[root] != root):
                    root = link_map[root]
                labels[row][col] = root

    json_map = []
    for label in np.unique(labels):
        if label != 0:
            temp = np.where(labels == label)
            x = int(np.min(temp[1]))
            y = int(np.min(temp[0]))
            w = int(np.max(temp[1]) - np.min(temp[1]))
            h = int(np.max(temp[0]) - np.min(temp[0]))
            json_map.append({"bbox": [x, y, w, h], "name": "UNKNOWN"})

    # Generating Image Zoning Features
    image_zone_vectors = []
    for index, item in enumerate(json_map):
        x, y, w, h = item["bbox"]
        img = np.array(data[y:y + h, x:x + w])
        if(img.shape[0] == 0 or img.shape[1] == 0):
            continue
        if(img.shape[0] >= img.shape[1]):
            img = cv2.resize(img, (round((scaled_size * img.shape[1]) / img.shape[0]), scaled_size), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (scaled_size, round((scaled_size * img.shape[0]) / img.shape[1])), interpolation=cv2.INTER_NEAREST)

        # Calculating zoning vector from grid size zones
        # Overlays the image onto fixed scaled template
        temp_img = np.ones((scaled_size, scaled_size), dtype=np.uint8) * 255
        x_lower, y_lower = temp_img.shape[0] // 2 - img.shape[0] // 2, temp_img.shape[1] // 2 - img.shape[1] // 2
        x_higher, y_higher = temp_img.shape[0] // 2 + img.shape[0] // 2, temp_img.shape[1] // 2 + img.shape[1] // 2
        if (img.shape[0] % 2 != 0):
            x_lower -= 1
        if (img.shape[1] % 2 != 0):
            y_lower -= 1
        temp_img[x_lower: x_higher, y_lower: y_higher] = img
        img = temp_img
        del temp_img

        # Calculating pixel count for each zone
        zone_vector = []
        zone_shape = (img.shape[0] // grid_size[0], img.shape[1] // grid_size[1])  # eg 9,8
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                img_zone = img[i * zone_shape[0]:(i * zone_shape[0] + zone_shape[0]), j * zone_shape[1]:(j * zone_shape[1] + zone_shape[1])]
                img_zone = np.where(img_zone <= 100, 1, 0)
                zone_vector.append(int(np.sum(img_zone)))
        image_zone_vectors.append(zone_vector)

    with open(os.path.join('features', 'zoning_features.json'), "r") as file:
        zoning_features = json.load(file)

    zoning_features['image_zone_vectors'] = image_zone_vectors

    with open(os.path.join('features', 'zoning_features.json'), "w") as file:
        json.dump(zoning_features, file)

    return json_map


def recognition(json_map, characters):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    # Test Image Stats
    # line 1 - 28, line 2 - 27, line 3 - 23, line 4 - 20, line 5 - 24, line 6 - 21
    # Total: 143

    # For 2
    # 4, 80, 123, 124 #num: 4
    # For a
    # 13, 16, 38, 48, 73, 75*, 112, 140, 141 #num: 9
    # For c
    # 24, 69, 89, 105 #num: 4
    # For dot
    # 53, 54, 95, 139 #num: 4
    # For e
    # 20, 26, 39, 40, 49, 50, 52, 77, 94, 107, 109, 113, 117, 118 #num: 14

    # Total: 35
    with open(os.path.join('features', 'zoning_features.json'), "r") as file:
        zoning_map = json.load(file)
        template_zone_vectors = zoning_map['template_zone_vectors']
        image_zone_vectors = zoning_map['image_zone_vectors']

    for i, image_zv in enumerate(image_zone_vectors):
        cos_sims = []
        for j, template_zv in enumerate(template_zone_vectors):
            num = np.sum(np.multiply(image_zv, template_zv))
            denum = np.multiply(np.sqrt(np.sum(np.square(image_zv))), np.sqrt(np.sum(np.square(template_zv))))
            cos_sims.append(num / denum)
        max_index = np.argmax(cos_sims)
        if (cos_sims[max_index] >= 0.92):
            json_map[i]["name"] = characters[max_index][0]

    return json_map


def save_results(results, rs_directory):
    """
    Donot modify this code
    """
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
    results = ocr(test_img, characters)
    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
