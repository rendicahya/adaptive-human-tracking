import cv2
import numpy as np


def to_rect(keypoints):
    if len(keypoints) < 1:
        return []

    # index = map(lambda m:m.trainIdx, matches)
    # kps = map(lambda i:train_kp[i], index)

    x = map(lambda kp: kp.pt[0], keypoints)
    y = map(lambda kp: kp.pt[1], keypoints)

    xmin = min(x)
    ymin = min(y)

    return [int(i) for i in [xmin, ymin, max(x) - xmin, max(y) - ymin]]


def spatial_filter(keypoints, kx, ky):
    x = map(lambda kp: kp.pt[0], keypoints)
    y = map(lambda kp: kp.pt[1], keypoints)

    xmean = np.mean(x)
    ymean = np.mean(y)

    xstdev = np.std(x)
    ystdev = np.std(y)

    xrange = kx * xstdev
    yrange = ky * ystdev

    keypoints = filter(lambda kp:
                       xmean - xrange < kp.pt[0] < xmean + xrange and
                       ymean - yrange < kp.pt[1] < ymean + yrange
                       , keypoints)

    return keypoints
