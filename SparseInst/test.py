from my_demo import demo
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("123.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)
predictions, visualized_output = demo.run_on_image(
                img, 0.5)
a = predictions["instances"]._fields["pred_masks"]
b = predictions["instances"]._fields["pred_classes"]
mask = np.ones_like(img, dtype=np.bool8)
for i in range(len(b)):
    print(b[i].item())
    if(b[i].item() == 0):
        mask[a[i]] = False
img[mask] = 255
img[mask == False] = 0
img = np.concatenate((np.zeros_like(img),img[:,:,0, np.newaxis]),-1)
cv2.imwrite("tt.png", img)
# img = cv2.imread("./1534750222.png", cv2.IMREAD_UNCHANGED)
# img[:,:100,-1] = 255
# print(img[:,:,-1])
# a = np.array(img, dtype=np.uint8)
# ret, png = cv2.imencode(".png", a)
# png.tobytes()
# cv2.imshow("test", a)
# cv2.waitKey(0)
# print(img.shape)
# mydpi=96
# plt.clf()
# plt.figure(figsize=[img.shape[1]/mydpi,img.shape[0]/mydpi],dpi=mydpi)
# plt.imshow(img)
# print(img.shape[0], img.shape[1])
# # plt.axis('off')   # 去坐标轴
# # plt.xticks([])    # 去 x 轴刻度
# # plt.yticks([])    # 去 y 轴刻度
# plt.axis('off')
# fig = plt.gcf()
# fig.set_size_inches(img.shape[1]/mydpi,img.shape[0]/mydpi) #dpi = 300, output = 700*700 pixels

# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# plt.margins(0,0)

# plt.savefig("test.svg",format='svg',bbox_inches='tight', dpi=mydpi,pad_inches = 0)
# import PIL.Image as Image
#     # draw the renderer
# visualized_output.canvas.draw()
 
# # Get the RGBA buffer from the figure
# w, h = visualized_output.canvas.get_width_height()
# buf = np.fromstring(visualized_output.canvas.tostring_argb(), dtype=np.uint8)
# buf.shape = (w, h, 4)
# # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
# buf = np.roll(buf, 3, axis=2)
# image = Image.frombytes("RGBA", (w, h), buf.tostring())
# image = np.asarray(image)[:,:,::-1]
# cv2.waitKey(0)
# data = np.fromstring(visualized_output.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(visualized_output.canvas.get_width_height()[::-1] + (3,))
# print(data)

# cv2.imshow("test", visualized_output.fig)
# cv2.waitKey(0)
# visualized_output.img
# visualized_output.save("test.jpg")
        


