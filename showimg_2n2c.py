import cv2 as cv

img_path = "./output/2n2c/00839.jpg"
img_rgb = cv.imread(img_path)
cv.imshow("Detection Result", img_rgb)
cv.waitKey(0)
cv.destroyAllWindows()
