JOINT_MAP = {
    "MidHip": 0,
    "LHip": 1,
    "LKnee": 4,
    "LAnkle": 7,
    "LFoot": 10,
    "RHip": 2,
    "RKnee": 5,
    "RAnkle": 8,
    "RFoot": 11,
    "LShoulder": 16,
    "LElbow": 18,
    "LWrist": 20,
    "LHand": 22,
    "RShoulder": 17,
    "RElbow": 19,
    "RWrist": 21,
    "RHand": 23,
    "spine1": 3,
    "spine2": 6,
    "spine3": 9,
    "Neck": 12,
    "Head": 15,
    "LCollar": 13,
    "Rcollar": 14,
    "Nose": 24,
    "REye": 25,
    "LEye": 26,
    "REar": 27,
    "LEar": 28,
    "LHeel": 31,
    "RHeel": 34,
}

SMPL_IDX = range(24)

AMASS_JOINT_MAP = {
    "MidHip": 0,
    "LHip": 1,
    "LKnee": 4,
    "LAnkle": 7,
    "LFoot": 10,
    "RHip": 2,
    "RKnee": 5,
    "RAnkle": 8,
    "RFoot": 11,
    "LShoulder": 16,
    "LElbow": 18,
    "LWrist": 20,
    "RShoulder": 17,
    "RElbow": 19,
    "RWrist": 21,
    "spine1": 3,
    "spine2": 6,
    "spine3": 9,
    "Neck": 12,
    "Head": 15,
    "LCollar": 13,
    "Rcollar": 14,
}

AMASS_IDX = range(22)
AMASS_SMPL_IDX = range(22)

# Canonical SMPL-X joint blocks used by block-wise observation fitting.
# These are used as default targets when callers provide explicit body/hand/face
# keypoint blocks (dict input) to the public APIs.
SMPLX_BODY_IDX = range(22)
SMPLX_LEFT_HAND_IDX = range(25, 46)
SMPLX_RIGHT_HAND_IDX = range(46, 67)
SMPLX_FACE_IDX_START = 67
