# import tensorflow as tf
# import pandas as pd
# import numpy as np
#
# import data.creation.tfr_utils as tfr_utils
# import data.creation.image_processing as image_processing
# import data.creation.io_utils as io_utils
#
#
# def _create_temporal_dataset(label_filename):
#     with tf.gfile.GFile(label_filename, mode='r') as labels_csv:
#         tfr_files = []
#         labels = pd.read_csv(labels_csv, delimiter=';', header=None)
#         _, num_columns = labels.shape
#         labels.columns = ['video_id', 'sentence']
#         data_gen = _generate_labels_frames(labels)
#
#         for row in data_gen:
#             row_idx, video_id, flows = row
#
#         jpg_flow_x = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
#                       flows[..., 0]]
#         jpg_flow_y = [tf.compat.as_bytes(image_processing.encode_jpg(flow).tobytes()) for flow in
#                       flows[..., 1]]
#
#         context = tf.train.Features(feature={'video_id': tfr_utils.int64_feature(video_id)})
#         feature_list = {
#             'label': tfr_utils.float_list_feature_list(embedded_sentence),
#             'label_ids': tfr_utils.float_list_feature_list(np.expand_dims(np.asarray(sentence_ids), axis=-1)),
#             'out_layers': tfr_utils.bytes_feature_list(jpg_frames),
#             'flow_x': tfr_utils.bytes_feature_list(jpg_flow_x),
#             'flow_y': tfr_utils.bytes_feature_list(jpg_flow_y)
#         }
#
#
# def _generate_labels_flows(labels):
#     frame_files = io_utils.get_frame_files(video_id, frames_path, frames_max)
#     # row_idx, video_id, flows =
#     return row_idx, video_id, flows
#
# def _get_flows(video_id, frames_path_frames_max)
