from ultralytics import YOLO

model = YOLO('./model/YOLO/best.pt')

def pred(img):
  (h, w) = img.size
  h = int(h)
  w = int(w)

  result = model.predict(img, save=False, imgsz=640, conf=0.25)
  box_tensor = result[0].boxes
  all_box = box_tensor.boxes.tolist()

  all_box_detect = []
  index = 1
  for i in all_box:
      x, y, x1, y1 = int(i[0]) - 5, int(i[1]) - 5, int(i[2])+5, int(i[3])+5      
      # image_crop = img[y: y1, x : x1 ]
      image_crop = img.crop((x, y, x1, y1))
      filename = 'img_' + str(index) + '.jpg'

      image_crop.save(filename)
      all_box_detect.append(filename)
      index += 1
  
  return all_box_detect