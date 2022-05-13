import cv2
import numpy as np
import reference as ref
import finder as find
import os

IN_DIR = "img/striped"
FIN_DIR = "test/striped"


def remove_shadow(img):

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    # cv2.imwrite('shadows_out_norm.jpg', result_norm)
    return result_norm
    # cv2.imwrite('shadows_out.jpg', result)
    # cv2.imwrite('shadows_out_norm.jpg', result_norm)

#**********************#


def can():
    img = cv2.imread("shadows_out_norm.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 200, apertureSize=3)
    cv2.imwrite("Canny.png", edges)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3), (-1, -1))
    dilated = cv2.dilate(edges, element)
    cv2.imwrite("Eroded.png", dilated)

    minLineLength = 200
    maxLineGap = 5

    lines = cv2.HoughLinesP(dilated, cv2.HOUGH_PROBABILISTIC, np.pi/180, 150, minLineLength,
                           maxLineGap)

    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            pts = np.array([[x1, y1], [x2, y2]], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0))

    cv2.imwrite('dilate_final.png', img)


# retrieves contours from img1, then takes from img2
# also rewrites into final directory
def retrieve(img, shapes, img2):
    img_contour = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 5)
    bkg_level = img_blur[int(5)][int(5)]
    thresh_level = bkg_level + ref.THRESH_C
    retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_bw, ref.CANNY_THRESH, ref.CANNY_THRESH/2)
    img_dilated = cv2.dilate(img_canny, kernel=np.ones((5, 5), np.uint8), iterations=2)

    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_stack = ref.stack_images(1, ([img, img_gray, img_blur],
                                      [img_bw, img_contour, img_dilated]))

    card_imgs = {}
    names = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > ref.CARD_MIN_AREA:
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
                width, height = ref.CARD_WIDTH, ref.CARD_HEIGHT
                new = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
                p1s = np.float32(ref.format_points(approx))
                # p1s = np.float32(approx)
                warped_canny = ref.warp(img_canny, p1s, new, width, height)

                #warped = ref.warp(img, p1s, new, width, height)
                warped = ref.warp(img2, p1s, new, width, height)

                # do name matching
                card = ref.Card()
                card.img = warped
                name = ref.match(card, shapes)
                card.name = name
                names.append(name)
                card_imgs[name] = card.img
                # card_imgs.append(card.img)

                cv2.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 3)
                print("name:", name)
                cv2.putText(img_contour, ref.name_from_id(name),
                            (x + (w // 20), y + (h // 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, ref.FONT_SIZE,
                            (0, 0, 0), 5)
            else:
                obj_type = ""
    cv2.imshow("Stack", img_stack)
    cv2.waitKey(ref.WAIT_TIME)
    return card_imgs, names, img_contour


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
                warped = remove_shadow(warped)
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


def match(card, shapes):
    best_shape_match_diff = 100000
    best_shape_name = "tbd"
    name = "placeholder"
    if len(card.img) != 0:
        # Difference the query card shapes from each shapes image; store the result with the least difference
        for shape in shapes:
            # print(len(card.img.shapes))
            # print(len(shapes.img.shapes))
            img_gray = cv2.cvtColor(card.img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (17, 17), 5)
            bkg_level = img_gray[10][10]
            # print(bkg_level)
            thresh_level = bkg_level-30
            retval, img_bw = cv2.threshold(img_blur, thresh_level, 255, cv2.THRESH_BINARY)


            # show_wait("cycle", img_bw, 25)

            diff_img = cv2.absdiff(img_bw, shape.img)
            shape_diff = int(np.sum(diff_img) / 255)
            cv2.imshow(ref.stack_images(0.8, ([img_bw],
                                              [diff_img])))
            cv2.waitKey(5000)

            if shape_diff < best_shape_match_diff:
                best_shape_match_diff = shape_diff
                #print("jello")
                best_shape_name = shape.name[0] + "_0" + shape.name[3]
                # print(best_shape_match_diff, best_shape_name)
    return name

# processes & isolates cards from images
# shadow removed, inside fill "removed"
def process(gs=True):
    for filename in os.listdir(IN_DIR):
        if filename.endswith("." + ref.IMG_TYPE):
            path = IN_DIR + "/" + filename
            print(path)
            img = cv2.imread(path)

            img_contour = img.copy()
            #img_contour = remove_shadow(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (27, 27), 3)
            img_canny = cv2.Canny(img_blur, 50, 50)
            img_dilated = cv2.dilate(img_canny, kernel=np.ones((5, 5), np.uint8), iterations=2)
            #img = img_contour.copy()
            # get_contours(filename, img, img_contour, img_gray, img_blur, img_canny, img_dilated, gs)
            get_contours(filename, img, img_contour, img_gray, img_blur, img_canny, img_dilated, gs)
            img_stack = ref.stack_images(0.8, ([img, img_gray, img_blur],
                                               [img_canny, img_contour, img_dilated]))
            cv2.imshow("Stack", img_stack)
            cv2.waitKey(3000)


def run():
    shapes = ref.load_shapes("shapes/")
    path = "test/IMG_0548.JPG"
    image = cv2.imread(path)
    #image = cv2.imread("shadows_out.jpg")
    img2 = remove_shadow(path)
    imgs, names, output = retrieve(image, shapes, img2)
    num_cards = len(imgs)
    print(names)
    sets = find.solve(names)
    print(sets)
    print(find.convert_sets(sets))
    # stack = ref.stack_images(1, get_vals(imgs))
    # cv2.imshow("processed", stack)
    cv2.imshow("output", output)
    find.draw_sets(imgs, sets)
    cv2.waitKey(0)


def matching():
    card = ref.Card()
# run()
process()