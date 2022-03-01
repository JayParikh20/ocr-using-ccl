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

# TODO
import sys
large_width = 400
np.set_printoptions(linewidth=large_width)
np.set_printoptions(threshold=sys.maxsize)


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
    # TODO
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
    # TODO Add your code here. Do not modify the return and input arguments

    # TODO
    template_max_height = 32
    grid_size = (16, 8)

    template_zone_vectors = enrollment(characters, grid_size, template_max_height)
    json_map = detection(test_img)

    image_zone_vectors = []
    for index, item in enumerate(json_map):
        x, y, w, h = item["bbox"]
        img = np.array(test_img[y:y + h, x:x + w])
        img = cv2.resize(img, (round((template_max_height * img.shape[1]) / img.shape[0]), template_max_height), interpolation=cv2.INTER_NEAREST)
        # Calculating zoning vector  from 3 x 3 area
        # adds white rows and cols as long as it is divisible by 3
        while (img.shape[0] % grid_size[0] != 0):
            img = np.append(img, np.ones((1, img.shape[1]), dtype=np.uint8) * 255, axis=0)
        while (img.shape[1] % grid_size[1] != 0):
            img = np.append(img, np.ones((img.shape[0], 1), dtype=np.uint8) * 255, axis=1)

        # Calculating pixel count for each zone
        zone_vector = []
        zone_shape = (img.shape[0] // grid_size[0], img.shape[1] // grid_size[1])  # eg 9,8
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                img_zone = img[i * zone_shape[0]:(i * zone_shape[0] + zone_shape[0]), j * zone_shape[1]:(j * zone_shape[1] + zone_shape[1])]
                img_zone = np.where(img_zone <= 100, 1, 0)
                zone_vector.append(np.sum(img_zone))
        image_zone_vectors.append(zone_vector)

    results = recognition(template_zone_vectors, image_zone_vectors, json_map, characters, test_img)
    return results


def enrollment(characters, grid_size, template_max_height):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    template_zone_vectors = []
    for char in characters:
        img: np.ndarray = char[1]

        # Trimming white borders
        del_cols = np.where(np.mean(img, axis=0) > 245)
        img = np.delete(img, del_cols, axis=1)
        del_rows = np.where(np.mean(img, axis=1) > 245)
        img = np.delete(img, del_rows, axis=0)
        img = cv2.resize(img, (round((template_max_height*img.shape[1])/img.shape[0]), template_max_height), interpolation=cv2.INTER_NEAREST)

        # Calculating zoning vector  from 3 x 3 area
        # adds white rows and cols as long as it is divisible by 3
        while (img.shape[0] % grid_size[0] != 0):
            img = np.append(img, np.ones((1, img.shape[1]), dtype=np.uint8)*255, axis=0)
        while (img.shape[1] % grid_size[1] != 0):
            img = np.append(img, np.ones((img.shape[0], 1), dtype=np.uint8)*255, axis=1)

        # Calculating pixel count for each zone
        zone_vector = []
        zone_shape = (img.shape[0] // grid_size[0], img.shape[1] // grid_size[1])     # eg 9,8
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                img_zone = img[i*zone_shape[0]:(i*zone_shape[0] + zone_shape[0]), j*zone_shape[1]:(j*zone_shape[1] + zone_shape[1])]
                img_zone = np.where(img_zone <= 100, 1, 0)
                zone_vector.append(np.sum(img_zone))
        template_zone_vectors.append(zone_vector)

    return template_zone_vectors


def detection(data):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    # TODO: Step 2 : Your Detection code should go here.

    def search(tree, child):
        if child != tree[child]:
            tree[child] = search(tree, tree[child])
        return tree[child]

    def merge(tree, child, parent):
        c, p = search(tree, child), search(tree, parent)
        if p != c:
            tree[c] = p

    bw_threshold = 100

    image = np.array(data)
    image = np.where(image <= bw_threshold, 0, 255)
    link_map = []
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
                            link_map.append((neighbors[0], neighbors[1]))

    linked = [i for i in range(label_counter)]
    for i, j in set(link_map):
        merge(linked, i, j)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (image[row][col] < 255):
                labels[row][col] = search(linked, labels[row][col])

    json_map = []
    for label in np.unique(labels):
        if label != 0:
            temp = np.where(labels == label)
            x = int(np.min(temp[1]))
            y = int(np.min(temp[0]))
            w = int(np.max(temp[1]) - np.min(temp[1]))
            h = int(np.max(temp[0]) - np.min(temp[0]))
            json_map.append({"bbox": [x, y, w, h], "name": "UNKNOWN"})

    return json_map


def recognition(template_zvs: list, image_zvs: list, json_map, characters, data):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    # Test Image Stats
    # line 1 - 28, line 2 - 27, line 3 - 23, line 4 - 20, line 5 - 24, line 6 - 21
    # Total: 143

    # For 2
    # 4, 80, 123, 124 #num: 4
    # For a
    # 13, 16, 38, 48, 73, 75*, 112, 140, 141 #num: 9
    # For dot
    # 53, 54, 95, 139 #num: 4
    # For e
    # 20, 26, 39, 40, 49, 50, 52, 77, 94, 107, 109, 113, 117, 118 #num: 14
    # For c
    # 24, 69, 89, 105 #num: 4
    # Total: 35

    for i, image_zv in enumerate(image_zvs):
        cos_sims = []
        for j, template_zv in enumerate(template_zvs):
            num = np.sum(np.multiply(image_zv, template_zv))
            denum = np.multiply(np.sqrt(np.sum(np.square(image_zv))), np.sqrt(np.sum(np.square(template_zv))))
            cos_sims.append(num/denum)
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
