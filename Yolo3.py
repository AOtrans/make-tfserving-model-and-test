from __future__ import division

import os
import cv2
import sys

import numpy as np
from tensorflow.keras import backend as K
#import tensorflow.keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from PIL import Image, ImageFont, ImageDraw

from model import yolo_eval, yolo_body

sys.path.append("..")
import CONFIG
def get_inputs(param):
    my_input = tf.map_fn(elems=param, fn=tf.image.decode_jpeg, dtype=tf.uint8)
    my_input = K.reshape(my_input,[-1, 416, 416, 3])
    my_input = K.cast(my_input, dtype=tf.float32)
    my_input = tf.divide(my_input, tf.constant(255.0))
    return my_input

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, _ = image.shape
    w, h = size
    print type(ih),type(iw),type(w),type(h)
    scale = min(w / iw, h / ih)
    print scale
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.ones((416, 416, 3), dtype=np.int8) * 128
    new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
  
    return new_image


class YOLO(object):
    def __init__(self):

        self.graph =tf.Graph()
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config, graph=self.graph)
        #KTF.set_session(session)
        K.set_session(session)
	#K._LEARNING_PHASE = tf.constant(0)
	K.set_learning_phase(0)

        self.model_path = CONFIG.yolo3_weigths_path
        self.anchors_path = CONFIG.yolo3_anchors_path
        self.classes_path = CONFIG.yolo3_classes_path
        self.score = CONFIG.yolo3_score_threshold
        self.iou = CONFIG.yolo3_iou_threshold
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
	

        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.font = ImageFont.truetype(font=CONFIG.font_path,
                                       size=np.floor(25.0).astype('int32'))
        with self.graph.as_default():
            self.boxes, self.scores, self.classes = self.generate()
	
    def outputs(self):
	return [self.boxes, self.scores, self.classes]

    def inputs(self):
	return [self.base_input,self.input_image_shape]

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

	my_input = Input(shape=[],dtype=tf.string)
	self.base_input = my_input
	my_input = Lambda(get_inputs, output_shape=(416,416,3))(my_input)
	my_input = Input(tensor=my_input)
        self.yolo_model = yolo_body(my_input, num_anchors // 3, num_classes)
        self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match

        print('Detection model, {} model, {} anchors, and {} classes load success!.'.format(model_path, num_anchors, num_classes))

        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def get_box(self, image):
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        with self.graph.as_default():
        	out_boxes, out_scores, out_classes = self.sess.run(
            	[self.boxes, self.scores, self.classes],
            	feed_dict={
                	self.yolo_model.input: image_data,
                	self.input_image_shape: [image.shape[0], image.shape[1]],
                	K.learning_phase(): 0
            		})

        return out_boxes, out_scores, out_classes

    def draw_image(self, image):

        out_boxes, out_scores, out_classes = self.get_box(image)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        image = Image.fromarray(image)
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, self.font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='red')
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill='red')
            draw.text(text_origin, label, fill=(0, 0, 0), font=self.font)
            del draw
        return image

    def close_session(self):
        self.sess.close()
