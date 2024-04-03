def Morph(license_plate_crop):
   # license_plate_detector
    results = license_plate_detector(license_plate_crop,augment=True)
    pr=results.pred[0]
    boxes=pr[:,:4]
    boxes=boxes.cpu().numpy()
    cropped = license_plate_crop.crop((boxes[0][0],boxes[0],[1],boxes[0][2],boxes[0][3]))
    cropped = cropped.crop((cropped.wodth*0.05,0,cropped.width,cropped.height))
    temp_file="temp.png"
    cropped.save(temp_file)
    plate=cv2.imread(temp_file)
    plate=cv.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    roi = cv2.threshold(plate,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cleared = clear_border(roi)
    cv2.imwrite("clear_border.png",cleared)
    cv