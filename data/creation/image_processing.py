import cv2
import numpy as np

import constants.constants as const
import constants.hyper_params as hp


def load_image(source):
    from tensorflow.python.lib.io import file_io
    if const.GCLOUD:
        with file_io.FileIO(source, mode='rb') as image_stream:
            img = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source)
    return img


def load_opt_flow_image(source):
    from tensorflow.python.lib.io import file_io
    if const.GCLOUD:
        with file_io.FileIO(source, mode='rb') as image_stream:
            img = cv2.imdecode(np.asarray(bytearray(image_stream.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    return img


def crop_and_pad_image(height, width, img):
    img = crop_img(height, width, img)
    img = pad_img(height, width, img)
    img = img.astype(np.float32)
    return img


def crop_img(height, width, img):
    """
    Crops image to target height and width. If input is smaller, leave original size.
    :param height: target height
    :param width: target width
    :param img: input image as type np.ndarray
    :return: a cropped np.ndarray
    """
    img_height, img_width, n_channels = img.shape
    height_dif = max(img_height - height, 0)
    height_start = int(np.floor(height_dif / 2))
    height_stop = height_start + height
    width_dif = max(img_width - width, 0)
    width_start = int(np.floor(width_dif / 2))
    width_stop = width_start + width

    return img[height_start: height_stop, width_start:width_stop]


def pad_img(height, width, img):
    """
    Pad image to target height and width with zeros. Image is set in the middle, e.g.:
                           0 0 0 0 0 0 0
        1 1 1 1            0 1 1 1 1 0 0
        1 1 1 1     -->    0 1 1 1 1 0 0
                           0 0 0 0 0 0 0
                           0 0 0 0 0 0 0

    :param height: target height
    :param width: target width
    :param img: input image as type np.ndarray
    :return: a np.ndarray with the target shape padded with zeros
    """
    img_height, img_width, n_channels = img.shape
    if img_height > height or img_width > width:
        print("Image bigger than padded target")
        return None
    else:
        new_img = np.zeros((height, width, n_channels), img.dtype)
        h_upper_border = int((height - img_height) / 2)
        w_left_border = int((width - img_width) / 2)
        h_lower_border = h_upper_border
        w_right_border = w_left_border
        if (height - img_height) % 2 == 1:
            h_lower_border += 1
        if (width - img_width) % 2 == 1:
            w_right_border += 1
        new_img[h_upper_border:height - h_lower_border, w_left_border:width - w_right_border, :] = img
        return new_img


def shift_channels(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_jpg(img):
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), hp.JPG_QUAL])[1]


def create_tvl1_dense(frames, skip=hp.SKIP, bound=hp.BOUND):
    """
    Creates optical flow frames for input frames. First flow frame is from frame[0] to frame[0],
    so input frames and output flow frames got the same length.
    """
    flows = []
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    prev_path = frames[0]
    prev_img = load_image(prev_path)
    height, width, _ = prev_img.shape
    low_prev_img = rescale_frame(prev_img, width//hp.FLOW_RESIZE_FACTOR, height//hp.FLOW_RESIZE_FACTOR)
    prev_gray = cv2.cvtColor(low_prev_img, cv2.COLOR_RGB2GRAY)

    for i in range(0, len(frames), skip):
        path = frames[i]
        img = load_image(path)
        height, width, _ = img.shape
        low_img = rescale_frame(img, width//hp.FLOW_RESIZE_FACTOR, height//hp.FLOW_RESIZE_FACTOR)
        gray = cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY)
        flow = optical_flow.calc(prev_gray, gray, None)
        flow = np.round((flow + bound) / (2. * bound) * 255.)
        flow[flow < 0] = 0
        flow[flow > 255] = 255
        flows.append(flow)
        prev_gray = gray

    return flows


def flip_img(img):
    return cv2.flip(img, 0)


def rescale_frame(frame, width, height):
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


