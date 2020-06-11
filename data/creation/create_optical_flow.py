import constants.constants as const
import data.creation.image_processing as image_processing
import data.creation.io_utils as io_utils
import constants.hyper_params as hp

import os
import cv2

if __name__ == '__main__':
    video_folder = const.SOMETHING_VIDEO_BASE_PATH
    horiz_flow_dir = os.path.join(const.SOMETHING_OPT_FLOW_BASE_PATH, 'u')
    vert_flow_dir = os.path.join(const.SOMETHING_OPT_FLOW_BASE_PATH, 'v')
    if not os.path.exists(horiz_flow_dir):
        os.makedirs(horiz_flow_dir)
    if not os.path.exists(vert_flow_dir):
        os.makedirs(vert_flow_dir)
    for d in os.listdir(video_folder):
        u_base_dir = os.path.join(horiz_flow_dir, d)
        v_base_dir = os.path.join(vert_flow_dir, d)
        if not os.path.exists(u_base_dir):
            os.makedirs(u_base_dir)
        else:
            continue
        if not os.path.exists(v_base_dir):
            os.makedirs(v_base_dir)
        frames = io_utils.get_frame_files(d, video_folder, 999999)
        flows = image_processing.create_tvl1_dense(frames, skip=1)
        for i, flow in enumerate(flows):
            u, v = cv2.split(flow)
            file_name = os.path.split(frames[i])[1]
            cv2.imwrite(os.path.join(u_base_dir, file_name), u, [int(cv2.IMWRITE_JPEG_QUALITY), hp.JPG_QUAL])
            cv2.imwrite(os.path.join(v_base_dir, file_name), v, [int(cv2.IMWRITE_JPEG_QUALITY), hp.JPG_QUAL])

