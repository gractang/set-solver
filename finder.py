import cv2
import reference as ref
import numpy as np


# returns whether or not the three cards are a set
def is_set(c1, c2, c3):
    # checks whether x is same or different
    # if mod 3 = 0, then python interprets as false. not --> turn to true
    # n = num, c = color, f = fill, s = shape
    n_sd = not ((int(c1[0]) + int(c2[0]) + int(c3[0])) % 3)
    c_sd = not ((int(c1[1]) + int(c2[1]) + int(c3[1])) % 3)
    f_sd = not ((int(c1[2]) + int(c2[2]) + int(c3[2])) % 3)
    s_sd = not ((int(c1[3]) + int(c2[3]) + int(c3[3])) % 3)
    return n_sd and c_sd and f_sd and s_sd


# a lovely n^3 solution haha
def solve(board):
    num_cards = len(board)
    sets = set()  # lol look it's a set! hahahahah
    for first_card in range(num_cards - 2):
        for second_card in range(first_card + 1, num_cards - 1):
            for third_card in range(second_card + 1, num_cards):
                c1 = board[first_card]
                c2 = board[second_card]
                c3 = board[third_card]

                if is_set(c1, c2, c3):
                    sets.add((c1, c2, c3))
    return sets


# converts sets of numbers into words (ex 0000 --> 1 red solid oval)
def convert_sets(sets):
    new = set()
    for s in sets:
        new.add((ref.name_from_id(s[0]), ref.name_from_id(s[1]), ref.name_from_id(s[2])))
    return new


# draws the sets out by stacking rows of three
def draw_sets(imgs, sets):
    stacks = []
    for s in sets:
        ims = [imgs[s[0]], imgs[s[1]], imgs[s[2]]]
        stacks.append(ims)
    mega_stack = ref.stack_images(.7, stacks)
    cv2.imshow("sets!", mega_stack)


# returns an array that contains the values in a dictionary
def get_vals(dict):
    arr = []
    for pair in dict:
        arr.append(dict[pair])
    return arr


def run():
    shapes = ref.load_shapes("shape/")
    image = cv2.imread("test/IMG_3667.jpg")
    #image = cv2.imread("shadows_out.jpg")
    imgs, names, output = ref.retrieve(image, shapes)
    num_cards = len(imgs)
    print(names)
    sets = solve(names)
    print(sets)
    print(convert_sets(sets))
    # stack = ref.stack_images(1, get_vals(imgs))
    # cv2.imshow("processed", stack)
    cv2.imshow("output", output)
    draw_sets(imgs, sets)
    cv2.waitKey(0)


if __name__ == '__main__':
    run()
