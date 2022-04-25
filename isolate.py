# Takes a raw image of a card, including its background, and isolates it

import cv2
import numpy as np
import reference as ref
import os

DIRECTORY = "img"
FIN_DIR = "all"


def get_contours(name, img, img_contour, img_gray, img_blur, img_canny, img_dilated, gs):
    # contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:

            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)

            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 4:
                objectType = "Card"
                width, height = ref.CARD_WIDTH, ref.CARD_HEIGHT
                # new = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                new = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                p1s = np.float32(ref.format_points(approx))
                warped_canny = ref.warp(img_canny, p1s, new, width, height)
                warped = None
                if gs:
                    (thresh, img_bw) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
                    warped = ref.warp(img_bw, p1s, new, width, height)
                else:
                    warped = ref.warp(img, p1s, new, width, height)
                # warped_dilated = cv2.dilate(warped, kernel=np.ones((3, 3), np.uint8), iterations=1)
                cv2.imshow("warped1", warped_canny)
                cv2.imshow("warped orig", warped)
                cv2.imwrite(FIN_DIR + "/" + name, warped)
            else:
                objectType = ""

            cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_contour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 255), 2)
    img_stack = ref.stack_images(0.8, ([img, img_gray, img_blur],
                                      [img_canny, img_contour, img_dilated]))
    cv2.imshow("Stack", img_stack)


# gs = grayscale, actually is black/white
# final directory must already exist
def process(gs=True):
    for filename in os.listdir(DIRECTORY):
        if filename.endswith("." + ref.IMG_TYPE):
            print(os.path.join(DIRECTORY + "/", filename))
            img = cv2.imread(DIRECTORY + "/" + filename)
            img_contour = img.copy()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (7, 7), 3)
            img_canny = cv2.Canny(img_blur, 50, 50)
            img_dilated = cv2.dilate(img_canny, kernel=np.ones((5, 5), np.uint8), iterations=2)
            get_contours(filename, img, img_contour, img_gray, img_blur, img_canny, img_dilated, gs)
            img_stack = ref.stack_images(0.8, ([img, img_gray, img_blur],
                                              [img_canny, img_contour, img_dilated]))
            cv2.imshow("Stack", img_stack)
            cv2.waitKey(1000)


process()

# img = np.zeros((ref.CARD_HEIGHT, ref.CARD_WIDTH, 3), np.uint8)
# img[:] = 0, 0, 0
# cv2.imwrite("reference/black.jpg", img)
