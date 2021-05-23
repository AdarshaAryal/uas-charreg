import cv2

imagePath = 'img002-00180.png'
img = cv2.imread(imagePath)
img = cv2.resize(img, (0,0), fx=0.05, fy=0.05)
cv2.imwrite('new'+imagePath,img)
cv2.destroyAllWindows()


