## here is implement REST API
import cv2
import sys
import time
import json
import yaml
import unidecode
import numpy as np
from copy import deepcopy
from src.OCR.detection.text_detector import TextDetector
from src.OCR.recognition.text_recognizer import TextRecognizer
from src.OCR.recognition.vn_text_recognizer import VNTextRecognizer
from src.utils.utility import sorted_boxes, padding_box, get_text_image, get_rotate_crop_image, preprocess_text, find_xy_center, draw_boxes_ocr, draw_result_image, draw_lines_ocr, extract_json_from_string

with open("./config/doc_config.yaml", "r") as f:
    doc_config = yaml.safe_load(f)
STATUS = doc_config["status"]

import os
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

logging.getLogger().setLevel(logging.ERROR)
os.environ['CURL_CA_BUNDLE'] = ''

class ocr_parser():
    __instance__ = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if ocr_parser.__instance__ == None:
            ocr_parser()
        return ocr_parser.__instance__
    
    def __init__(self):
        if ocr_parser.__instance__ != None:
            raise Exception("Parser is a singleton!")
        else:
            ocr_parser.__instance__ == self
        self.lang = "en"
        self.en_recognizer = TextRecognizer.getInstance()
        self.vn_recognizer = VNTextRecognizer.getInstance()
        self.recognizer = self.vn_recognizer 
        self.detector = TextDetector.getInstance()


    def extract_info(self, image, is_visualize=True):
        """
        Match template with input image
        Input: input image
        Output: result image, result json file
        """
        status_code = "200"
        with open('src/result/grv_tem.json') as yaml_file:
            result = yaml.safe_load(yaml_file)
        # Detection
        detection = self.detector.detect(image)

        if detection is None:
            status_code = '460'
            return result, status_code
        detection = sorted_boxes(detection)
        n_boxes = len(detection)

        # Recognition
        recognition = []
        for i in range(n_boxes):
            box = detection[i]
            box = padding_box(image, box, max_pixel=1, left_side=0.001, right_side=0.001, top_side=0.001, bottom_side=0.001)
            box = np.array(box, dtype=np.float32).copy()
            cropped_image = get_rotate_crop_image(image, box)
            text = self.recognizer.recognize(cropped_image)
            text = preprocess_text(text)
            recognition.append([box.tolist(), text])
        return recognition



if __name__ == "__main__":
    start_time = time.time()
    img_path = "/home/bqthinh/Documents/images/GRV 1.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    prescription = ocr_parser()
    recognition = prescription.extract_info(image)
    print(recognition)
    print(img_path)
    print(time.time() - start_time)