import cv2
import numpy as np

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
#模型路径和预测图片路径
path_model="/content/drive/MyDrive/yolov5-5.0/yolov5-master/yolov5s.onnx"
image_path="/content/drive/MyDrive/yolox-pytorch-main/img/street.jpg"
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
#check_requirements(('opencv-python>=4.5.4',))
#模型加载
net = cv2.dnn.readNetFromONNX(path_model)
#加载图片及变换[1,3,640,640]
images=cv2.imread(image_path)
image_shape=images.shape[:2]
image=images.copy()
new_unpad=(640,640)
image = cv2.resize(image,new_unpad, interpolation=cv2.INTER_LINEAR)
image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
image = np.ascontiguousarray(image)
image=image[None]
image=image/255.0
#模型推理
net.setInput(image)
pred = net.forward()

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
  print(len(unique_class))
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
for i in output:
  images=cv2.rectangle(images,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0),2)
  images=cv2.putText(images,"%s:%.2f"%(class_names[int(i[-1])],i[-2]),(int(i[0]),int(i[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
cv2.imwrite("test2.jpg",images)
 
