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
large_width = 400
np.set_printoptions(linewidth=large_width)


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
        characters_list: list of characters along with name for each character.

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
    # enrollment()
    # show_ground(test_img)
    # return
    json_map = detection(test_img)

    results = recognition(test_img, json_map, characters)

    # with open("results.json", "w") as write_file:
    #     json.dump(json_map, write_file)
    return results

    # TODO
    # raise NotImplementedError


# TODO
def show_ground(data):
    color = (0, 0, 0)
    ground = np.array(data)
    with open("data/groundtruth.json") as file:
        groundtruth = json.load(file)

    for j, item in enumerate(groundtruth):
        x, y, w, h = item["bbox"]
        cv2.rectangle(ground, (x, y), (x+w, y+h), color, 1)
    show_image(ground, delay=10000)

    res = np.array(data)
    with open("results.json") as file:
        results = json.load(file)

    for j, item in enumerate(results):
        x, y, w, h = item["bbox"]
        cv2.rectangle(res, (x, y), (x+w, y+h), color, 1)
    show_image(res, delay=10000)

def enrollment():
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    # TODO
    # raise NotImplementedError


def detection(data):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    bw_threshold = 100

    # image = np.array(data[54:108, 4:92])
    image = np.array(data)
    image = np.where(image <= bw_threshold, 0, 255)
    # print(image)
    # show_image(image, delay=2000)
    '''image = np.array([
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    image[image == 0] = 255'''
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
                    if (neighbors == np.zeros((2,))).any():  # any one neighbor has no label
                        labels[row][col] = np.max(neighbors)
                    else:
                        labels[row][col] = np.min(neighbors)
                        if (neighbors[0] != neighbors[1]):  # both neighbors are not same, but are linked
                            link_map.append((neighbors[0], neighbors[1]))

    # print(f"linked map: {set(link_map)}")
    linked = [i for i in range(label_counter)]
    for i, j in set(link_map):
        union(linked, i, j)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (image[row][col] < 255):
                labels[row][col] = find(linked, labels[row][col])

    json_map = []
    color = (0, 0, 0)
    for label in np.unique(labels):
        # print(np.where(labels == label)[0])
        if label != 0:
            # print("For label:", label)
            temp = np.where(labels == label)
            # cv2.rectangle(data, (np.min(temp[1]), np.min(temp[0])), (np.max(temp[1]), np.max(temp[0])), color, 1)
            # bbox.append((np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])))
            x = int(np.min(temp[1]))
            y = int(np.min(temp[0]))
            w = int(np.max(temp[1]) - np.min(temp[1]))
            h = int(np.max(temp[0]) - np.min(temp[0]))
            json_map.append({"bbox": [x, y, w, h], "name": "UNKNOWN"})
    # num = 0
    # print(labels[bbox[num][0]:bbox[num][1], bbox[num][2]:bbox[num][3]]*255)
    # show_image(data, delay=10000)

    # print("counter: ", label_counter)
    # print("map: ", linked)

    # print(len(np.unique(labels)))
    # print(np.unique(labels))
    # show_image(labels, delay=5000)
    # TODO
    # raise NotImplementedError
    return json_map


def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]


def union(data, i, j):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pi] = pj


def connected(data, i, j):
    return find(data, i) == find(data, j)


def recognition(data, json_map, characters):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    # x, y, w, h = json_map[84]["bbox"]
    # print(data[y:y + h, x:x + w])
    # return

    # Test Image Stats
    # line 1 - 28
    # line 2 - 27
    # line 3 - 23
    # line 4 - 20
    # line 5 - 24
    # line 6 - 21
    # Total: 143

    # For 2 best 75 with BW
    # 4
    # 80
    # 123
    # 124

    # For a best 65 with BW
    # 13
    # 16
    # 38
    # 48
    # 73
    # 75*
    # 112
    # 140
    # 141

    # For dot best 70-90 with BW, i-53, 54, 95, 139
    # 53
    # 54
    # 95
    # 139

    # For e best 73 with BW
    # 20
    # 26
    # 39
    # 40
    # 49
    # 50
    # 52
    # 77
    # 94
    # 107
    # 109
    # 113
    # 117
    # 118

    # For c best .92 with BW
    # 24
    # 69
    # 89
    # 105
    # ncc_map = []
    # print(f"for char: {characters[2][0]}")

    for i, char in enumerate(characters):
        # characters[0][1][1:-1, 4:-5] #2
        # characters[1][1][1:-1, 2:-4] #a
        # characters[2][1][2:-2, 1:-3] #c
        # characters[3][1][3:-2, 2:-5] #dot
        # characters[4][1][1:-3, 1:-4] #e
        print(f"Loop for {char[0]}: ")
        template = np.array(char[1])
        # curr_ncc_map = []
        for j, item in enumerate(json_map):
            x, y, w, h = item["bbox"]
            image = np.array(data[y:y + h, x:x + w])
            # if(j == 141):
            # print(data[y:y+h, x:x+w])
            # print(data[y:y+h, x:x+w])
            # 2: index 4
            # a: index 13
            # x, y, w, h = json_map[4]["bbox"]
            # print(data[y:y+h, x:x+w])
            image = np.where(image <= 100, 0, 255)
            template = np.where(template <= 100, 0, 255)
            # get the maximum of both sizes
            max_size = (np.max([template.shape[0], image.shape[0]]), np.max([template.shape[1], image.shape[1]]))
            # maps the smaller image onto bigger
            centered_image = np.ones(max_size) * 255
            centered_template = np.ones(max_size) * 255
            image_min = np.subtract(max_size, image.shape) // 2
            image_max = np.add(max_size, image.shape) // 2
            template_min = np.subtract(max_size, template.shape) // 2
            template_max = np.add(max_size, template.shape) // 2
            # sets the values
            centered_image[image_min[0]: image_max[0], image_min[1]: image_max[1]] = image
            centered_template[template_min[0]:template_max[0], template_min[1]:template_max[1]] = template
            # if(i == 53 or i == 54 or i == 95 or i == 143):
            #     print(centered_image)
            #     print("||||")
            #     print(centered_template)
            # releases unused memory
            del image

            # NCC
            numerator = np.sum(np.multiply(centered_image, centered_template))
            denominator = np.sqrt(np.multiply(np.sum(np.square(centered_image)), np.sum(np.square(centered_template))))
            if ((numerator / denominator) > 0.60):
                # curr_ncc_map.append(j)
                # print(j)
                json_map[j]["name"] = str(char[0])
    return json_map
    # PMCC
    # image_mean = np.mean(centered_image)
    # template_mean = np.mean(centered_template)
    # image_std = np.std(centered_image)
    # template_std = np.std(centered_template)
    # print(image_mean)
    # print(image_std, "\n")
    # summation = np.sum(np.multiply((centered_image - image_mean), (centered_template - template_mean)))
    # if summation == 0 and (image_std == 0 or template_std == 0):
    #     corr = 1000
    # else:
    #     corr = np.divide(summation, (np.std(centered_image) * np.std(centered_template)))
    # corrs.append(corr)
    # print(f"{i}: {corr}")
    # mean_corr = np.mean(corrs)
    # max_corr = np.max(corrs) * 0.9
    # print(max_corr)
    # for i, corr in enumerate(corrs):
    #     if (corr > max_corr):
    #         print(i)
    # for irow in range(image.shape[0]):
    #     for icol in range(image.shape[1]):
    #         CCorr = 0
    #         for trow in range(template.shape[0]):
    #             for tcol in range(template.shape[1]):
    #                 CCorr += abs(image[irow][icol]- template[trow][tcol])
    # print(CCorr)

    # TODO
    # raise NotImplementedError


# TODO
def save_results(results, rs_directory):
    """
    Donot modify this code
    """
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    # TODO
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
    results = ocr(test_img, characters)
    # TODO
    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
