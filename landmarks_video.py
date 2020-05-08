from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils, dlib, cv2, time, json
  
LANDMARK_MODEL = "./models/shape_predictor_68_face_landmarks.dat"
# dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL)

def eye_aspect_ratio(eye):
	# calculates eye aspect ratio
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)
	return EAR

EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 2

COUNTER = 0 # counts frames
TOTAL = 0 # counts total blinks

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]




# starting the camera
vs = VideoStream(src=0).start()
time.sleep(2.0)

start = time.time()
ear_data = {'time': [], 'ear': []}

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)

	
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0 # the eye aspect ratio

		ear_data['time'].append(time.time()-start)
		ear_data['ear'].append(ear)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			COUNTER = 0

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# mark the landmarks
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	cv2.imshow("EDITED", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
 
cv2.destroyAllWindows()
vs.stop()

# dump data into json.file
data_file = open("./data/ear_data.json", "w") 
json.dump(ear_data, data_file, indent = 6) 
data_file.close() 
