import cv2
import numpy as np
import os


WAIT_TIME = 5000

# all possible shape formats
SHAPES = ["0__0", "0__1", "0__2", "1__0", "1__1", "1__2", "2__0", "2__1", "2__2"]
IMG_TYPE = "jpg"

# assigns values for each number, color, fill, and shape
NUM_DICT = {0: "1", 1: "2", 2: "3"}
COLOR_DICT = {0: "red", 1: "green", 2: "purple"}
FILL_DICT = {0: "solid", 1: "striped", 2: "empty"}
SHAPE_DICT = {0: "oval", 1: "diamond", 2: "squiggle"}

# threshold for canny
CANNY_THRESH = 10
THRESH_C = 70

# min and max area for something to be considered a card
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 5000

# width and height for card to be scaled to
# actual card dimensions: 2.25 x 3.5 in
CARD_WIDTH = 225
CARD_HEIGHT = 350

# for visualization purposes
FONT_SIZE = 2

# when fill is unknown, use default
DEFAULT_FILL = 0

# range values for red, green, and purple as determined by hand using color.py
RED_VALS = [0, 49, 64, 255, 212, 255]
GREEN_VALS = [50, 89, 70, 255, 56, 255]
PURPLE_VALS = [90, 179, 12, 255, 82, 255]
VALS_DICT = {0:RED_VALS, 1:GREEN_VALS, 2:PURPLE_VALS}


# idk why i did this. this is the same as the shape class lol
class Card:
    def __init__(self):
        self.name = []
        self.img = []


# really just a class to represent the shape images
class Shape:
    def __init__(self):
        self.img = []
        self.name = "tbd"

    def __init__(self, image, n):
        self.img = image
        self.name = n

    def __repr__(self):
        return name_from_id(self.name, True)


# shows image for time (ms)
def show_wait(img_name, img, time):
    cv2.imshow(img_name, img)
    # waits (ms). 0 = forever
    cv2.waitKey(time)


# gets name from number id (id as string like 0000)
def name_from_id(id):
    return NUM_DICT[int(id[0])] + " " + COLOR_DICT[int(id[1])] + " " + FILL_DICT[int(id[2])] + " " + SHAPE_DICT[int(id[3])]


# pts is (or should be) an np.ndarray of size 4x1x2
# like so
# [[[1597   65]]
#
#  [[1552  946]]
#
#  [[2111  974]]
#
#  [[2158   87]]]
# where each element is [[xn yn]]
# so xn is pts[n][0][1]
def format_points(pts):
    # if incorrect input, return
    if len(pts) != 4:
        return
    left_top = [0,0]
    left_bottom = [0,0]
    right_bottom = [0,0]
    right_top = [0,0]
    x_sum = 0
    y_sum = 0

    # find averages of all x and y points (i.e. midpoint of rectangle)
    for pt in pts:
        x_sum += pt[0][0]
        y_sum += pt[0][1]
    x_avg = x_sum // 4
    y_avg = y_sum // 4

    for pt in pts:
        if pt[0][0] <= x_avg:
            if pt[0][1] <= y_avg:
                left_top = pt
            else:
                left_bottom = pt
        else:
            if pt[0][1] <= y_avg:
                right_top = pt
            else:
                right_bottom = pt
    return [left_top, left_bottom, right_bottom, right_top]


# warps image (pts1 --> pts2)
def warp(img, pts1, pts2, width, height):
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    output = cv2.warpPerspective(img, matrix, (width, height))
    return output


# empty function for trackbars
def empty(dummy):
    print(dummy)


# stacks images in img_array (2-d array of images), scaling by scale value
def stack_images(scale, img_array):
    rows = len(img_array)

    # if asked to stack nothing, return default black image
    if rows == 0:
        return cv2.imread("reference/black.jpg")
    cols = len(img_array[0])

    # checks to make sure that img_array does actually contain rows in a list format
    rows_available = isinstance(img_array[0], list)

    # get width and height of each image (assumed to be identical?)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        # hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)

    # i guess if there's only one row? i.e. img_array is 1d
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


# returns color id of given card object
def match_color(card):
    image = card.img
    img_results = []
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(3):
        color_vals = VALS_DICT[i]
        lower = np.array([color_vals[0], color_vals[2], color_vals[4]])
        upper = np.array([color_vals[1], color_vals[3], color_vals[5]])
        mask = cv2.inRange(img_hsv, lower, upper)
        img = cv2.bitwise_and(image, image, mask=mask)
        img_results.append(img)

    # stack = stack_images(1, img_results)
    # show_wait("maps", stack,WAIT_TIME)
    # cv2.imwrite("test/maps.jpg", stack)

    color = 0
    black = cv2.imread("reference/black.jpg")
    # want most different
    best_match_diff = -1
    for i in range(len(img_results)):
        color_name = COLOR_DICT[i]
        # print(color_name)
        diff_img = cv2.absdiff(black, img_results[i])
        diff = int(np.sum(diff_img) / 255)
        # print(diff)

        if diff > best_match_diff:
            best_match_diff = diff
            color = i

    return color


def match(card, shapes):
    best_shape_match_diff = 100000
    best_shape_name = "tbd"
    name = "placeholder"
    if len(card.img) != 0:
        # Difference the query card shape from each shape image; store the result with the least difference
        for shape in shapes:
            # print(len(card.img.shape))
            # print(len(shape.img.shape))
            img_gray = cv2.cvtColor(card.img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (7, 7), 5)
            bkg_level = img_gray[10][10]
            # print(bkg_level)
            thresh_level = bkg_level-30
            retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY)

            # show_wait("cycle", img_bw, 25)

            diff_img = cv2.absdiff(img_bw, shape.img)
            shape_diff = int(np.sum(diff_img) / 255)
            if shape_diff < best_shape_match_diff:
                best_shape_match_diff = shape_diff
                #print("jello")
                best_shape_name = shape.name[0] + "_0" + shape.name[3]
                # print(best_shape_match_diff, best_shape_name)
        color_id = match_color(card)
    name = best_shape_name[0] + str(color_id) + str(DEFAULT_FILL) + best_shape_name[3]
    return name


# loads shape into array of Shape objects
def load_shapes(dir_in):
    shapes = []
    for filename in os.listdir(dir_in):
        if filename.endswith("." + IMG_TYPE):
            #print(os.path.join(DIRECTORY + "/", filename))
            img = cv2.imread(dir_in + "/" + filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shapes.append(Shape(img_gray, filename[:-4]))
    return shapes


# takes in input image & shapes list
# returns list of isolated images,
# the names of those images (in id form),
# and the drawn-over image
def retrieve(img, shapes):
    img_contour = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 5)
    bkg_level = img_blur[int(5)][int(5)]
    thresh_level = bkg_level + THRESH_C
    retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_bw, CANNY_THRESH, CANNY_THRESH/2)
    img_dilated = cv2.dilate(img_canny, kernel=np.ones((5, 5), np.uint8), iterations=2)
    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_stack = stack_images(1, ([img, img_gray, img_blur],
                                      [img_bw, img_contour, img_dilated]))
    card_imgs = {}
    names = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > CARD_MIN_AREA:
            # draw onto img_contour, all the contours, draw all (-1), draw in red (0,0,255), line thickness 10
            cv2.drawContours(img_contour, cnt, -1, (0, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            num_corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            name = ""
            if num_corners == 4:
                # do card identification & warping
                width, height = CARD_WIDTH, CARD_HEIGHT
                new = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                p1s = np.float32(format_points(approx))
                # p1s = np.float32(approx)
                warped_canny = warp(img_canny, p1s, new, width, height)
                warped = warp(img, p1s, new, width, height)

                # do name matching
                card = Card()
                card.img = warped
                name = match(card, shapes)
                card.name = name
                names.append(name)
                card_imgs[name] = card.img
                # card_imgs.append(card.img)

                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 3)
                print("name:", name)
                cv2.putText(img_contour, name_from_id(name),
                            (x + (w // 20), y + (h // 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,
                            (0, 0, 0), 5)
            else:
                obj_type = ""



    cv2.imshow("Stack", img_stack)
    cv2.waitKey(WAIT_TIME)
    return card_imgs, names, img_contour
