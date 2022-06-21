import cv2
import numpy as np
import reference as ref


def empty(dummy):
    print(dummy)


h_min, h_max = 0, 179
s_min, s_max = 0, 255
v_min, v_max = 0, 255

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 840, 240)
cv2.createTrackbar("Hue Min", "TrackBars", h_min, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", h_max, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", s_min, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", s_max, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", v_min, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", v_max, 255, empty)


vid = cv2.VideoCapture(1)
vid.set(3, 640)
vid.set(4, 480)
try:
    while True:
        success, img = vid.read()
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        #print(h_min, h_max, s_min, s_max, v_min, v_max)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Original",img)
        # cv2.imshow("HSV",imgHSV)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Result", imgResult)

        img_stack = ref.stack_images(0.6, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("TrackBars", img_stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except:
    print("uh")
print(h_min, h_max, s_min, s_max, v_min, v_max)