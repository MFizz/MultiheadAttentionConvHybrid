import tensorflow as tf
import os
import random


def get_frame_files(video_id, frames_path, frames_max, skip=1):
    frame_path = os.path.join(frames_path, str(video_id))
    files = tf.gfile.ListDirectory(frame_path)
    sorted_files = sorted(files)
    skipped_files = sorted_files[::skip]
    num_frames = len(skipped_files)
    if frames_max < num_frames:
        sample_indices = sorted(random.sample(range(num_frames), frames_max))
        frames = [sorted_files[j] for j in sample_indices]
    else:
        frames = sorted_files
    frame_files = [os.path.join(frame_path, file_name) for file_name in frames]

    return frame_files


def get_opt_flow_files(video_id, jpg_path, opt_flow_path, frames_max, skip, bracket_size):
    frame_path = os.path.join(jpg_path, str(video_id))
    files = tf.gfile.ListDirectory(frame_path)
    sorted_files = sorted(files)
    # start jpg_frame so tha opt_flow starts at 1. frame
    start = bracket_size // 2
    skipped_files = sorted_files[start::skip]
    if frames_max < len(skipped_files):
        indices = sorted(random.sample(range(start, len(files), skip), frames_max))
        frames = [sorted_files[j] for j in indices]
    else:
        indices = list(range(start, len(files), skip))
        frames = skipped_files
    jpg_frame_files = [os.path.join(frame_path, file_name) for file_name in frames]

    u_frame_path = os.path.join(opt_flow_path, 'u', str(video_id))
    v_frame_path = os.path.join(opt_flow_path, 'v', str(video_id))
    u_opt_flow_files = tf.gfile.ListDirectory(u_frame_path)
    sorted_u_opt_flow_files = sorted(u_opt_flow_files)
    brackets = get_opt_flow_brackets(sorted_u_opt_flow_files, u_frame_path, v_frame_path, indices, bracket_size)

    return jpg_frame_files, brackets


def get_opt_flow_brackets(frames, u_path, v_path, selected_frame_indices, stack_size):
    bracket_ids = []
    u_files = []
    v_files = []
    assert stack_size < len(frames)
    old_stop = -1
    for frame in selected_frame_indices:
        start = frame - stack_size // 2
        stop = frame + (stack_size + 1) // 2
        if frame - stack_size // 2 <= 0:
            start = 0
            stop = stack_size
        elif frame + (stack_size + 1) // 2 > len(frames):
            start = len(frames) - stack_size
            stop = len(frames)
        if old_stop > start:
            start = old_stop
        u_files.extend([os.path.join(u_path, f) for f in frames[start:stop]])
        v_files.extend([os.path.join(v_path, f) for f in frames[start:stop]])
        bracket_ids.append((len(u_files) - stack_size, len(u_files)))
        old_stop = stop
    return bracket_ids, u_files, v_files

if __name__ == '__main__':
    import constants.constants as const
    jpg_path = const.SOMETHING_VIDEO_BASE_PATH
    opt_path = const.SOMETHING_OPT_FLOW_BASE_PATH
    get_opt_flow_files(1, jpg_path, opt_path, 99999, 10, 5)