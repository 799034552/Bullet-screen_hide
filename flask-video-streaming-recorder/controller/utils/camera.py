import cv2
import threading
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__) or __file__, "../../../SparseInst")))
from my_demo import demo
import cv2
from PIL import Image
from torch.cuda.amp import autocast
import numpy as np
import PIL.Image as Image


class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./static/video.avi', fourcc, 20.0, (640, 480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # 打开摄像头， 0代表笔记本内置摄像头
        self.cap = cv2.VideoCapture(2)
        print("运行到这里")

        # 初始化视频录制环境
        self.is_record = False
        self.out = None
        self.mask = None

        # 视频录制线程
        self.recordingThread = None

    # 退出程序释放摄像头
    def __del__(self):
        self.cap.release()
    
    def get_mask(self):
        return self.mask

    def get_frame(self):
        ret, frame = self.cap.read()
        # print(frame)
        # frame = cv2.imread("./aaa.png")
        # print(frame)
        # print(os.getcwd())
        # exit()
        img = cv2.resize(frame, dsize=(frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        # img = img.
        # exieeeresize([img.shape[0] // 2, img.shape[1] // 2, img.shape[2]])
        with autocast(enabled=True):
            predictions, visualized_output = demo.run_on_image(
                            img, 0.5)
            a = predictions["instances"]._fields["pred_masks"]
            b = predictions["instances"]._fields["pred_classes"]
            mask = np.ones_like(img, dtype=np.bool8)
            for i in range(len(b)):
                if(b[i].item() == 0):
                    mask[a[i]] = False
            img[mask] = 255
            img[mask == False] = 0
            img = np.concatenate((np.zeros_like(img),img[:,:,0, np.newaxis]),-1)
            img = cv2.resize(img, dsize=(img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            ret, jpeg = cv2.imencode('.png', img)
            self.mask = jpeg.tobytes()
        
    # draw the renderer
        # visualized_output.canvas.draw()
        
        # # Get the RGBA buffer from the figure
        # w, h = visualized_output.canvas.get_width_height()
        # buf = np.fromstring(visualized_output.canvas.tostring_argb(), dtype=np.uint8)
        # buf.shape = (w, h, 4)
        # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = np.roll(buf, 3, axis=2)
        # image = Image.frombytes("RGBA", (w, h), buf.tostring())
        # image = image.resize((image.size[0] * 2, image.size[1] *2))
        # frame = np.asarray(image)
        # img = np.array(visualized_output, dtype=np.uint8)
        # frame = img[:,:,::-1]
        # frame = visualized_output

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)

            # 视频录制
            if self.is_record:
                if self.out == None:
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.out = cv2.VideoWriter('./static/video.avi', fourcc, 20.0, (640, 480))

                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)
            else:
                if self.out != None:
                    self.out.release()
                    self.out = None

            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
