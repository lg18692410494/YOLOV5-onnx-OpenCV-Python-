#导入IE/OpenCV/numpy/time模块
#需要提前安装好openvino
from openvino.inference_engine import IECore, IENetwork
import cv2
import numpy as np
from time import time

import datetime
 

def iou(b1,B2):
    inter_rect_x1=np.maximum(b1[0],B2[:,0])
    inter_rect_y1=np.maximum(b1[1],B2[:,1])
    inter_rect_x2=np.minimum(b1[2],B2[:,2])
    inter_rect_y2=np.minimum(b1[3],B2[:,3])

    inter_area=np.maximum(inter_rect_x2-inter_rect_x1,0)*np.maximum(inter_rect_y2-inter_rect_y1,0)
    area_b1=(b1[2]-b1[0])*(b1[3]-b1[1])
    area_b2=(B2[:,2]-B2[:,0])*(B2[:,3]-B2[:,1])
    iou=inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)

    return iou
#配置推断计算设备，onnx文件路径，图片路径

DEVICE = 'CPU'
#模型路径
path_model="/project/ev_sdk/src/YOLOV5/yolov5s.onnx"
#待测图片路径
image_path= "/home/data/665/TYSN_vehicle_climbing_20210107_train_1042.jpg"


class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
'teddy bear', 'hair drier', 'toothbrush']
start = datetime.datetime.now()
#初始化插件，输出插件版本号
ie = IECore()
ver = ie.get_versions(DEVICE)[DEVICE]
print("{descr}: {maj}.{min}.{num}".format(descr=ver.description, maj=ver.major, min=ver.minor, num=ver.build_number))
    
#读取onnx模型文件
net = ie.read_network(model=path_model)
 
#准备输入输出张量
print("Preparing input blobs")
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = 1


#载入模型到AI推断计算设备
print("Loading IR to the plugin...")
exec_net = ie.load_network(network=net, num_requests=1, device_name=DEVICE)
 
#读入图片
n, c, h, w = net.inputs[input_blob].shape



images = cv2.imread(image_path) 
#执行推断计算
print("Starting inference in synchronous mode")

image_shape=images.shape[:2]
image=images.copy()
new_unpad=(640,640)
image = cv2.resize(image,new_unpad, interpolation=cv2.INTER_LINEAR)
image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
image = np.ascontiguousarray(image)
image=image[None]
image=image/255.0


res = exec_net.infer(inputs={input_blob: image})


# 处理输出
print("Processing output blob")
pred = res["output"]

#解码
#[x, y, w, h] to [x1, y1, x2, y2]
shape_boxes=np.zeros(pred[:,:,:4].shape)
shape_boxes[:,:,0]=pred[:,:,0]-pred[:,:,2]/2
shape_boxes[:,:,1]=pred[:,:,1]-pred[:,:,3]/2
shape_boxes[:,:,2]=pred[:,:,0]+pred[:,:,2]/2
shape_boxes[:,:,3]=pred[:,:,1]+pred[:,:,3]/2
pred[:,:,:4]=shape_boxes

output=[]
for i in range(pred.shape[0]):
  prediction=pred[i]
  mask=(prediction[:,4]>0.5)
  #初步筛选置信度大于0.5
  prediction=prediction[mask]
  class_conf,class_pred=np.expand_dims(np.max(prediction[:,5:],1),-1),np.expand_dims(np.argmax(prediction[:,5:],1),-1)
  #种类得分
  detections=np.concatenate((prediction[:,:5],class_conf,class_pred),1)

  unique_class=np.unique(detections[:,-1])
 
  best_box=[]
  #非极大抑制
  for c in unique_class:
    cls_mask=detections[:,-1]==c
    detection=detections[cls_mask]
    #同一种类
    arg_sort=np.argsort(detection[:,4])[::-1]
    detection=detection[arg_sort]
    #根据得分从大到小进行排序
    while np.shape(detection)[0] >0:
      best_box.append(detection[0])
      if len(detection) ==1:
        break
      #计算一个框和同种类后面所有的iou
      ious=iou(best_box[-1],detection[1:])    
      detection=detection[1:][ious<0.4]
      #   iou>0.4的被剔除
  output.append(best_box)
output=np.array(output)
output=np.squeeze(output,axis=0)

#还原原图比例  
output[:,0]*=image_shape[1]/new_unpad[0]
output[:,1]*=image_shape[0]/new_unpad[1]
output[:,2]*=image_shape[1]/new_unpad[0]
output[:,3]*=image_shape[0]/new_unpad[1]
#画图
start1 = datetime.datetime.now()
for i in output:
    images=cv2.rectangle(images,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
    images=cv2.putText(images,"%s:%.2f"%(class_names[int(i[-1])],i[-2]),(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
#保存绘画后图像
cv2.imwrite("test2.jpg",images)

print("Inference is completed")
print("时间差：",start1-start)
