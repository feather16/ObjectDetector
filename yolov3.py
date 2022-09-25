import core.utils as utils
from core.yolov3 import YOLOv3, decode

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os
import tkinter
import tkinter.filedialog

def request_file(url: str, path: str) -> None:
    file = requests.get(url, stream=True)
    with open(path, "wb") as f:
        f.write(file.content)

class Yolov3Model():
    INPUT_SIZE = 416
    def create(self):
        def extract_bbox_tensors(feature_maps):
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, i)
                bbox_tensors.append(bbox_tensor)
            return bbox_tensors

        # モデルの構築
        input_layer = tf.keras.layers.Input([self.INPUT_SIZE, self.INPUT_SIZE, 3])
        feature_maps = YOLOv3(input_layer)
        bbox_tensors = extract_bbox_tensors(feature_maps)
        self.model = tf.keras.Model(input_layer, bbox_tensors)

        data_dir = f'{os.path.dirname(__file__)}/data'
        weights_file_path = f"{data_dir}/yolov3.weights"
        if not os.path.exists(weights_file_path):
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            print('Downloading weights ...')
            request_file(weights_url, weights_file_path)
            print('Downloaded.')
        utils.load_weights(self.model, weights_file_path)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = tf.keras.models.load_model(path)

    def predict(self, image_path: str) -> Image:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(
            np.copy(original_image), [self.INPUT_SIZE, self.INPUT_SIZE]
        )
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(
            pred_bbox, original_image_size, self.INPUT_SIZE, 0.3
        )
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        return image

class GUI(tkinter.Frame):
    def __init__(self, master: tkinter.Tk=None):
        super().__init__(master)
        self.size = (450, 300)
        self.master = master
        self.master.geometry(f'{self.size[0]}x{self.size[1]}')
        self.master.resizable(0, 0)
        self.master.title('Object Detector')

        # ボタンの配置
        button = tkinter.Button(
            self.master, 
            text='Select Image File', 
            command=self.select_image_file,
            font=("", 20)
        )
        button.pack()
        button.place(
            x=120,
            y=110,
        )

        self.yolov3_model = Yolov3Model()
        self.yolov3_model.create()

    def select_image_file(self) -> None:
        file_path = tkinter.filedialog.askopenfilename(
            filetypes=[
                ("JPEG Image", ".jpg"),
                ("PNG Image", ".png"),
            ],
            initialdir=os.getcwd(),
        )
        if file_path == '':
            return
        image = self.yolov3_model.predict(file_path)
        image.show()

tk = tkinter.Tk()
gui = GUI(master=tk)
gui.mainloop()