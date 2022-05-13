import os

import cv2
import numpy as np

# assigns values for each number, color, fill, and shapes
NUM_DICT = {0: "1", 1: "2", 2: "3"}
COLOR_DICT = {0: "red", 1: "green", 2: "purple"}
FILL_DICT = {0: "solid", 1: "empty", 2: "striped"}
SHAPE_DICT = {0: "oval", 1: "diamond", 2: "squiggle"}

# threshold for canny
CANNY_THRESH = 50
THRESH_C = 70

# min and max area for something to be considered a card
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 5000

SHAPE_MIN_AREA = 6000
SHAPE_MAX_AREA = 14000

SHAPE_WIDTH = 160
SHAPE_HEIGHT = 80

# width and height for card to be scaled to
# actual card dimensions: 2.25 x 3.5 in
CARD_WIDTH = 225
CARD_HEIGHT = 350

# for visualization purposes
FONT_SIZE = 0.5

# when fill is unknown, use default
DEFAULT_FILL = 0

# range values for red, green, and purple as determined by hand using color.py
RED_VALS = [0, 49, 64, 255, 212, 255]
GREEN_VALS = [50, 89, 70, 255, 56, 255]
PURPLE_VALS = [90, 179, 12, 255, 82, 255]
VALS_DICT = {0:RED_VALS, 1:GREEN_VALS, 2:PURPLE_VALS}

WAIT_TIME = 3000

EMPTY_MAX = 50
STRIPED_MAX = 180


# card class to store name and image
class Card:

    def __init__(self, name_in, img_in, contours_in):
        self.name = name_in
        self.img = img_in
        self.contours = contours_in


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


# isolates cards from img
# return tuple of:
# - array of images (cards in color warped to rectangle, no other processing)
# - image with contours drawn on
def isolate_cards(img):
    # make copy of image to draw contours onto later
    img_contour = img.copy()

    # preprocess image to make contouring easier
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 5)
    bkg_level = img_blur[int(5)][int(5)]
    thresh_level = bkg_level + THRESH_C
    retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY)

    img_canny = cv2.Canny(img_bw, CANNY_THRESH, CANNY_THRESH / 2)
    img_dilated = cv2.dilate(img_canny, kernel=np.ones((5, 5), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_stack = stack_images(1, ([img, img_gray, img_blur],
                                 [img_bw, img_contour, img_dilated]))

    card_imgs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > CARD_MIN_AREA:
            # draw onto img_contour, all the contours, draw all (-1), draw in red (0,0,255), line thickness 5
            cv2.drawContours(img_contour, cnt, -1, (0, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)

            # make an approximation of the four corner points
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            num_corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # print(w)
            # print(h)
            if num_corners == 4:
                # do card identification & warping
                width, height = CARD_WIDTH, CARD_HEIGHT
                new = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                p1s = np.float32(format_points(approx))

                warped = warp(img, p1s, new, width, height)
                card_imgs.append(warped)

                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                obj_type = ""

    # cv2.imshow("Stack", img_stack)
    # cv2.waitKey(WAIT_TIME)

    return card_imgs, img_contour


# removes regions of shadow from img
# credit to Dan Masek from this stackoverflow post:
# https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
def remove_shadow(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        img_stack = [dilated_img, bg_img, diff_img, norm_img]
        # cv2.imshow("shdjkfl", np.hstack(img_stack))
        # cv2.waitKey(0)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # result and normalized result (i can't tell the difference tbh)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    return result_norm


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


# # returns shapes id of given card object
# # shapes is actually a list of cards that have specific shapes attributes
# def match_shape(card, shapes):
#     best_shape_match_diff = 100000
#     best_shape_name = "tbd"
#     name = "placeholder"
#     if len(card.img) != 0:
#         # Difference the query card shapes from each shapes image; store the result with the least difference
#         for shapes in shapes:
#             img_gray = cv2.cvtColor(card.img, cv2.COLOR_BGR2GRAY)
#             # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 2)
#             bkg_level = img_gray[10][10]
#             thresh_level = bkg_level-30
#             retval, img_bw = cv2.threshold(img_gray, thresh_level, 255, cv2.THRESH_BINARY)
#
#             # rms = remove_shadow(card.img)
#             # rms_gray = cv2.cvtColor(rms, cv2.COLOR_BGR2GRAY)
#             # retval, rms_bw = cv2.threshold(rms_gray, rms_gray[10][10] - 30, 255, cv2.THRESH_BINARY)
#
#             # show_wait("cycle", img_bw, 25)
#             diff_img = cv2.absdiff(img_bw, shapes.img)
#             # diff_img = cv2.absdiff(rms_bw, shapes.img)
#             shape_diff = int(np.sum(diff_img) / 255)
#             if shape_diff < best_shape_match_diff:
#                 best_shape_match_diff = shape_diff
#                 best_shape_name = shapes.name
#                 # cv2.imshow("best", shapes.img)
#                 # cv2.imshow("card", rms_bw)
#                 # cv2.waitKey(0)
#
#     return best_shape_name


def match_shape(card, shapes):
    if len(card.contours) < 1:
        return "wrong"
    c = card.contours[0]
    img_gray = cv2.cvtColor(card.img, cv2.COLOR_BGR2GRAY)
    bkg_level = img_gray[10][10]
    thresh_level = bkg_level - 30
    thresh, img_bw = cv2.threshold(img_gray, thresh_level, 255, cv2.THRESH_BINARY)
    mask = np.zeros(img_bw.shape, np.uint8)
    cv2.drawContours(mask, c, -1, 255, -1)
    cv2.fillPoly(mask, pts=[c], color=255)

    peri = cv2.arcLength(c, True)

    # make an approximation of the four corner points
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    cropped_img = mask[y:y + h, x:x + w]
    cropped_resize = cv2.resize(cropped_img, (SHAPE_WIDTH, SHAPE_HEIGHT))

    retval, img_bw = cv2.threshold(cropped_resize, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("inv", img_bw)

    best_shape_name = "_"
    best_shape_match_diff = 1000000
    for shape in shapes:
        img = shape[1]
        shape_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, shape_bw = cv2.threshold(shape_gray, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow("shape", shape_bw)
        diff_img = cv2.absdiff(img_bw, shape_bw)
        # cv2.imshow("sjfakl", diff_img)
        # cv2.waitKey(0)

        shape_diff = int(np.sum(diff_img) / 255)
        if shape_diff < best_shape_match_diff:
            best_shape_match_diff = shape_diff
            best_shape_name = shape[0]

    return best_shape_name


# retrieves the contours inside the card, adds to card.contours
def contour_shape(card):
    img = card.img
    img_contour = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 3)
    # bkg_level = img_blur[int(15)][int(15)]
    thresh_level = img_gray[5][5]-30
    retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY_INV)

    img_canny = cv2.Canny(img_bw, CANNY_THRESH, CANNY_THRESH / 2)
    img_dilated = cv2.dilate(img_canny, kernel=np.ones((3, 3), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_stack = stack_images(1, ([img, img_gray, img_blur],
                                 [img_bw, img_contour, img_dilated]))
    # show_wait("stack", img_stack, 0)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if SHAPE_MIN_AREA < area < SHAPE_MAX_AREA:
            # draw onto img_contour, all the contours, draw all (-1), draw in red (0,0,255), line thickness 2
            cv2.drawContours(img_contour, cnt, -1, (0, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)

            # make an approximation of the four corner points
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            num_corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # print("Area: ", area)
            # print('Width:', w)
            # print('Height:', w)
            # print('Box area:', w*h)
            # print('Fraction:', 1.0*area/(w*h))
            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            card.contours.append(cnt)

    return img_contour


# gets the grayscale mean of the inside of one shapes on the card
def get_mean(img, contours):
    if len(contours) < 1:
        print(len(contours))
        return "wrong"
    c = contours[0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bkg_level = img_gray[10][10]
    thresh_level = bkg_level - 30
    thresh, img_bw = cv2.threshold(img_gray, thresh_level, 255, cv2.THRESH_BINARY)
    mask = np.zeros(img_bw.shape, np.uint8)
    cv2.drawContours(mask, c, -1, 255, -1)
    cv2.fillPoly(mask, pts=[c], color=(255, 255, 255))
    masked = cv2.bitwise_and(img_bw, img_bw, mask=mask)
    # cv2.imshow("mask", mask)
    # cv2.imshow("masked img", masked)
    # cv2.waitKey(0)

    mean = cv2.mean(img_bw, mask=mask)
    gs_mean = mean[0]
    return gs_mean


# matches the fill of the card where 0=filled, 1=empty, 2=striped
def match_fill(card):
    mean = get_mean(card.img, card.contours)
    print(mean)
    if mean > STRIPED_MAX:
        return 1
    elif mean > EMPTY_MAX:
        return 2
    else:
        return 0


# the number of contours in the card is just the number of shapes
def match_number(card):
    # subtract 1 because 0:1, 1:2, etc. in dictionary
    return len(card.contours)-1


# loads shapes into array of Shape objects
def load_shapes(dir_in):
    shapes = []
    for filename in os.listdir(dir_in):
        # print(os.path.join(DIRECTORY + "/", filename))
        img = cv2.imread(dir_in + "/" + filename)
        shapes.append((filename[0], img))
    return shapes


# # only run once to generate the shapes1 to compare with
# def generate_shapes(dir_in, dir_out):
#     for filename in os.listdir(dir_in):
#         print(os.path.join(dir_in + "/", filename))
#         img = cv2.imread(dir_in + "/" + filename)
#         # iso_card_img = isolate_cards(img)[0][0]
#         # cv2.imshow("jsklf", iso_card_img)
#         # cv2.waitKey(10)
#         #img_gray = cv2.cvtColor(iso_card_img, cv2.COLOR_BGR2GRAY)
#         rms = remove_shadow(img)
#         img_gray = cv2.cvtColor(rms, cv2.COLOR_BGR2GRAY)
#         retval, img_bw = cv2.threshold(img_gray, img_gray[10][10]-30, 255, cv2.THRESH_BINARY)
#         show_wait("stack", np.hstack([img_gray, img_bw]), 10)
#         cv2.imwrite(dir_out + "/" + filename, img_bw)


# run only once: generates a cropped version of just the shape from img, writes to dir_out
def generate_shapes(name, img, dir_out):
    card = Card("", img, [])
    contour_shape(card)
    if len(card.contours) < 1:
        return "wrong"
    c = card.contours[0]
    peri = cv2.arcLength(c, True)

    # make an approximation of the four corner points
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(approx)
    cropped_img = img[y:y+h, x:x+w]
    cv2.imshow("cropped", cropped_img)

    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    bkg_level = img_gray[5][5]
    thresh_level = bkg_level - 30
    thresh, img_bw = cv2.threshold(img_gray, thresh_level, 255, cv2.THRESH_BINARY)

    cv2.imshow("bw", img_bw)
    cv2.waitKey(0)
    cv2.imwrite(dir_out + '/' + name + ".JPG", img_bw)


# also only runs once— crops every image in dir_in to just be
# one single card
def crop_imgs(dir_in):
    for filename in os.listdir(dir_in):
        img = cv2.imread(dir_in + "/" + filename)
        cards_imgs, img_contour = isolate_cards(img)
        cv2.imwrite(dir_in + "/" + filename, cards_imgs[0])


# combines matching shapes and color
# returns the card with the name corrected
def match(card, shapes):
    img_contour = contour_shape(card)
    # show_wait("contours", img_contour, 0)
    number = match_number(card)
    color = match_color(card)
    fill = match_fill(card)
    shape = match_shape(card, shapes)
    card.name = str(number) + str(color) + str(fill) + shape
    return name_from_id(card.name)


# identifies cards
def card_id():
    img = cv2.imread("test/IMG_0554.jpg")
    shapes = load_shapes("test/shapes1")
    cards_imgs, img_contour = isolate_cards(img)
    show_wait("contours", img_contour, 0)
    id_cards_imgs = []
    for card_img in cards_imgs:
        card = Card("", card_img)
        print("best shapes:")
        card = match(card, shapes)
        print("name:", card.name)
        cv2.putText(card.img, name_from_id(card.name), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE,
                    (0, 0, 0), 1)
        id_cards_imgs.append(card.img)
    show_wait("identified cards", np.hstack(id_cards_imgs), 0)


def run():
    # for filename in os.listdir("test/shapes_imgs"):
    #
    #     card = Card("", cv2.imread("test/shapes_imgs/" + filename), [])
    #     contour_shape(card)
    #     # contours, img_contour = contour_shape(card)
    #     # show_wait("cont", img_contour, 0)
    #     print(name_from_id(filename[0] + "0" + filename[2:4]))
    #     # print(get_mean(card.img, contours))
    #
    #     print("fill: ", match_fill(card))
    #     print("number: ", len(card.contours))
    #     print("*******")

    return
    # card = Card("", cv2.imread("test/shapes_imgs/2_02.JPG"), [])
    # shapes = load_shapes("test/shapes")
    # name = match(card, shapes)
    # print(name)
    # show_wait(name, card.img, 0)



run()

