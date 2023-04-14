# gear-teeth-defect-detection-YOLOv3

### Download YOLOv3 weights
```
$ wget https://pjreddie.com/media/files/yolov3.weights
```

### Annotation the images with `LabelImg`
* To Install LabelImg - 
```
git clone 

conda create -n LabelImg python=3.9.13

conda activate LabelImg
pip install labelimg
```
* Open LabelImg
```
# activate the conda environment
conda activate LabelImg

# cd to the cloned labelImg-Repo
ls
cd lebelImg-Repo
ls

# open labelImg tool
python labelImg.py
```
* Annotation procedure;


### convert annotated xml to csv_file
```
# For training-set
python xml_to_csv.py --path_to_xml /path/to/train_images_folder --path_to_csv /path/to/train_images_folder/annotation.csv

# For testing-set
python xml_to_csv.py --path_to_xml /path/to/test_images_folder --path_to_csv /path/to/test_images_folder/annotation.csv
```

### Create labelmap.pbtxt
/img_xml_data/[labelmap.pbtxt](https://github.com/ThuraTunScibotics/gear-teeth-defect-detection-YOLOv3/blob/main/img_xml_data/labelmap.pbtxt)


### Convert csv_file to annotation.txt file in a separate data folder(yolov3_data)
```
# For train_gear_annotations.txt
python prepare_data.py --path_to_images /path/to/train_images_folder --path_to_csv_annotations /path/to/train_images_folder/annotation.csv --path_to_save_output /yolov3_data/train

# For test_gear_annotation.txt
python prepare_data.py --path_to_images /path/to/test_images_folder --path_to_csv_annotations /path/to/test_images_folder/annotation.csv --path_to_save_output /yolov3_data/test
```

### Adding class names
/classes/[gear_teeth.names](https://github.com/ThuraTunScibotics/gear-teeth-defect-detection-YOLOv3/blob/main/classes/gear_teeth.names)

### References;

https://github.com/sniper0110/YOLOv3

https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3

### Changing the necessary parameters in configuration file
/core/[config.py](https://github.com/ThuraTunScibotics/gear-teeth-defect-detection-YOLOv3/blob/main/core/config.py)
Change the preparameter & hyperparameters for model training based on the machine being trained on;
* add the class name path ('/classes/gear_teeth.names') / __C.YOLO.CLASSES
* training annotation path ('/dataset/train_gear_annotations.txt') / __C.TRAIN.ANNOT_PATH
* training batch size (depending on GPU size) / __C.TRAIN.BATCH_SIZE
* training input size of neurons (depending on GPU size) / __C.TRAIN.INPUT_SIZE
* data augmentation (True or False) / __C.TRAIN.DATA_AUG
* initial and final learning rate / __C.TRAIN.LR_INIT, __C.TRAIN.LR_END
* numbers of epochs / __C.TRAIN.EPOCHS
* testing annotation path ('/dataset/test_gear_annotations.txt') / __C.TEST.ANNOT_PATH

**Note** Model accuracy & performance will be depending on some of the hyperparameters such as epochs, batch size, neuron sizes and learning rate.
