import cv2
import numpy as np
import keypoint_utils as kputils
from time import time
import serial

cap = cv2.VideoCapture(0)

cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print 'File cannot be opened'
    exit()

grabbed, query = cap.read()

if not grabbed:
    print 'Failed to open input'
    exit()

ser = serial.Serial('/dev/ttyUSB0')

test = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
rect = cv2.selectROI('Select ROI', query, False, False)

cv2.destroyAllWindows()

hamming_bf = cv2.BFMatcher(cv2.NORM_HAMMING)
norm_bf = cv2.BFMatcher(cv2.NORM_L2)

x, y, w, h = tuple([int(i) for i in rect])
query = query[y:y + h, x:x + w]
query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

methods = [{
    'label': 'ORB',
    'detector': cv2.ORB_create(),
    'match_method': hamming_bf
}, {
    'label': 'AKAZE',
    'detector': cv2.AKAZE_create(),
    'match_method': hamming_bf
}, {
    'label': 'KAZE',
    'detector': cv2.KAZE_create(),
    'match_method': norm_bf
}, {
    'label': 'BRISK',
    'detector': cv2.BRISK_create(),
    'match_method': hamming_bf
}, {
    'label': 'SIFT',
    'detector': cv2.xfeatures2d.SIFT_create(),
    'match_method': norm_bf
}, {
    'label': 'SURF',
    'detector': cv2.xfeatures2d.SURF_create(),
    'match_method': norm_bf
}]

for method in methods:
    method['query_desc'] = method['detector'].detectAndCompute(query_gray, None)[1]

methods = [method for method in methods if method['query_desc'] is not None]

for method in methods:
    t = time()
    method['detector'].detectAndCompute(test, None)
    method['score'] = float(len(method['query_desc'])) / time() - t

methods = sorted(methods, key=lambda method: len(method['query_desc']), reverse=True)
method_info = ['%s (%d)' % (method['label'], len(method['query_desc'])) for method in methods]
print 'Available methods: %s' % ', '.join(method_info)

f = 0

while True:
    grabbed, train = cap.read()

    if not grabbed:
        break

    train_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)
    height, width = train_gray.shape
    all_keypoints = []
    used_methods = []
    f += 1

    for method in methods:
        kp, train_desc = method['detector'].detectAndCompute(train_gray, None)
        matches = method['match_method'].knnMatch(method['query_desc'], train_desc, k=2)
        matches = [m for m, n in matches if m.distance < .85 * n.distance]
        kp = [kp[m.trainIdx] for m in matches]
        kp = kputils.spatial_filter(kp, 1, 1)

        all_keypoints.extend(kp)
        used_methods.append(method['label'])

        if len(all_keypoints) > 20:
            break

    xmean = int(np.mean([kp.pt[0] for kp in all_keypoints]))
    ymean = int(np.mean([kp.pt[1] for kp in all_keypoints]))

    if ymean < height / 2:
        if xmean < width / 3:
            print 'KIRI'
            ser.write('4')
        elif xmean < 2 * width / 3:
            ser.write('1')
            print 'LURUS'
        else:
            ser.write('3')
            print 'KANAN'
    else:
        ser.write('0')
        print 'STOP'

    draw = train
    # draw = cv2.drawKeypoints(train, all_keypoints, outImage=None)
    cv2.line(draw, (xmean, 0), (xmean, height), (0, 0, 255), 2)
    cv2.line(draw, (0, ymean), (width, ymean), (0, 0, 255), 2)

    # print '#%d Using %s' % (f, ', '.join(m for m in used_methods))
    cv2.imshow('Combined Keypoint Methods', draw)
    # cv2.imwrite('human3-output1/%d.jpg' % f, draw)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
