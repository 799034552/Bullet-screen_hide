# import cv2
import numpy as np
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# print(ret, frame)
# img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# img = np.array(img, dtype=np.uint8)
# img = img[:,:,::-1]
# cv2.imshow("test" ,img)
# cv2.waitKey(0)

img = np.ones((1024,1024,3))
img = img.resize((img.shape[0] // 2, img.shape[1] // 2, img.shape[2]))
print(img)
