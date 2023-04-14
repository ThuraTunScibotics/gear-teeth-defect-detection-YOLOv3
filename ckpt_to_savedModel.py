import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode

# To avoid 
# "W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at conv_ops.cc:1120 : 
# NOT_FOUND: No algorithm worked! Error messages"
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__=='__main__':

    INPUT_SIZE   = cfg.TEST.INPUT_SIZE
    NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))

    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("./yolov3").expect_partial()

    # To avoid
    # WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. 
    # `model.compile_metrics` will be empty until you train or evaluate the model.
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      #        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       #       metrics=['accuracy'])
    # model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred}, 
      #              metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='yolo_loss', 
    #             metrics=['accuracy'])
    # traced_model = tf.autograph.to_code(model)                
    # traced_model.save('SavedModel/YOLOv3_model')
    
    tf.function(autograph=True, experimental_compile=True)

    model.save('SavedModel/YOLOv3_model')