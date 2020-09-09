from imageai.Detection import ObjectDetection
import os
import pandas as pd
img='test'
for i in range(1,22):
    img_path=img+str(i)+'.png'
    img_dec=img+str(i)+'new.png'
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , img_path), output_image_path=os.path.join(execution_path , img_dec))

    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

# execution_path = os.getcwd()
# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 'test19.png'), output_image_path=os.path.join(execution_path , 'test19new.png'))

# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )




