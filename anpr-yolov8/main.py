from ultralytics import YOLO
import cv2
import pandas as pd
import OCR
from sort.sort import *
from OCR import get_car, read_license_plate, write_csv
from moviepy.editor import *
'''
model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("veh1.mp4")
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
vehicle_classes = ["car", "truck", "bus", "motorcycle"]
vehicle_detected = False  # Flag to indicate if a vehicle has been detected
while True:
    count += 1
    if count % 4 != 0:
        continue
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in vehicle_classes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            vehicle_detected = True
        break
    
 
    if vehicle_detected:
        # Call  number plate reader function here!!
        print("Vehicle detected! Initiating number plate reader...")
                vehicle_detected = False

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''

results = {}

mot_tracker = Sort()
path = r' Enter the abs file path here '
def framerate(self,path):
    filen = path
    clip = VideoFileClip(filen)
    rate = clip.fps
    #print("FPS : " + str(rate)
    return str(rate)




# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r't_2.2\runs\detect\train\weights\best.pt')

# load video 
cap = cv2.VideoCapture(path)

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    
    frame_nmr += framerate(path)
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                #license_plate_text, license_plate_text_score = reader_2_3(license_plate_crop_thresh)
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                
# write results
write_csv(results, './test.csv')
