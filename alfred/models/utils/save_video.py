import imageio
import os
import torch
import numpy as np
import cv2
SAVE_FOLDER_NAME = "eval_video"
font = cv2.FONT_HERSHEY_SIMPLEX
toptomLeftCornerOfText = (10, 30)
middleLeftCornerOfText = (10, 230)
bottomLeftCornerOfText = (10, 270)
fontScale = 0.7
fontColor = (255, 0, 0)
lineType = 2


def images_to_video(save_dir, video_name, images, depths, list_actions, goal_instr, fail_reason, fps=1):
    save_dir = os.path.join(save_dir, SAVE_FOLDER_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video_name += ".mp4"
    save_dir = os.path.join(save_dir, video_name)
    writer = imageio.get_writer(save_dir, fps=fps)
    for image, depth, action in zip(images, depths, list_actions):
        depth = np.expand_dims(depth, axis=2)
        depth = np.tile(depth, (1, 3))
        cat_image = np.concatenate([image, depth], axis=1)
        cv2.putText(cat_image, action,
                    toptomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(cat_image, fail_reason,
                    middleLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(cat_image, goal_instr,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        writer.append_data(cat_image)
    writer.close()
