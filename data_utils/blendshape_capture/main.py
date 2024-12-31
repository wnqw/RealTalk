# -*- coding:utf-8 -*-
import argparse
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scipy.signal import savgol_filter

def infer_bs_all(root_path):
    """
    Processes all subfolders in `root_path`, for each .mp4 video:
      1) Creates a subfolder 'img/'.
      2) Reads frames, runs mediapipe face_landmarker.
      3) Saves blendshape array to an .npy file.
      4) Removes the 'img/' folder.
    """

    # Setup mediapipe face landmarker
    base_options = python.BaseOptions(model_asset_path="./data_utils/blendshape_capture/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Recursively or one-level scan subfolders (one-level example)
    # for subfolder in sorted(os.listdir(root_path)):
    #   subfolder_path = os.path.join(root_path, subfolder)
    #   if not os.path.isdir(subfolder_path):
    #       continue
    #   # then process videos in subfolder_path
    #   # <the rest of code>

    # If you only want one-level, do:
    for subfolder in sorted(os.listdir(root_path)):
        subfolder_path = os.path.join(root_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Now iterate over .mp4 files in this subfolder
        for filename in sorted(os.listdir(subfolder_path)):
            if filename.endswith(".mp4"):
                mp4_path = os.path.join(subfolder_path, filename)
                npy_path = mp4_path.replace('.mp4', '_bs.npy')
                if os.path.exists(npy_path):
                    print(f"[INFO] {npy_path} already exists. Skipping: {filename}")
                    continue
                else:
                    print(f"[INFO] Processing video: {filename}")

                    # 1) Create /img folder
                    img_dir = os.path.join(subfolder_path, 'img')
                    if os.path.exists(img_dir):
                        shutil.rmtree(img_dir)
                    os.makedirs(img_dir, exist_ok=True)

                    # 2) Prepare for reading frames
                    cap = cv2.VideoCapture(mp4_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print("fps:", fps, "frame_count:", frame_count)

                    bs_array = np.zeros((frame_count, 52), dtype=np.float32)

                    # 3) For each frame: detect blendshapes
                    pbar = tqdm(total=frame_count, desc="Frames")
                    index = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Save the frame to a temporary image
                        temp_image_path = os.path.join(img_dir, "temp.png")
                        cv2.imwrite(temp_image_path, frame)

                        # Use mediapipe FaceLandmarker
                        image_mp = mp.Image.create_from_file(temp_image_path)
                        result = detector.detect(image_mp)
                        
                        if len(result.face_blendshapes) > 0:
                            # face_blendshapes_scores => 52 categories
                            face_blendshapes_scores = [
                                cat.score for cat in result.face_blendshapes[0]
                            ]
                            # The original code slices [1:] then appends 0, 
                            # but you can adapt as needed. We replicate the snippet:
                            blendshape_coef = np.array(face_blendshapes_scores)[1:]
                            blendshape_coef = np.append(blendshape_coef, 0)
                            bs_array[index] = blendshape_coef
                        else:
                            # If no face detected, fill with zeros or keep existing
                            pass

                        index += 1
                        pbar.update(1)

                    pbar.close()
                    cap.release()

                    # 4) Smooth with Savitzky-Golay
                    smoothed = np.zeros_like(bs_array)
                    for col_idx in range(bs_array.shape[1]):
                        smoothed[:, col_idx] = savgol_filter(bs_array[:, col_idx], 5, 3)

                    # 5) Save .npy
                    np.save(npy_path, smoothed)
                    print(f"[INFO] Saved blendshape to: {npy_path}, shape={smoothed.shape}")

                    # 6) Remove /img folder
                    shutil.rmtree(img_dir)
                    print(f"[INFO] Removed temp folder: {img_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Root path containing subfolders with .mp4 files.")
    args = parser.parse_args()

    infer_bs_all(args.path)


# # -*-coding:utf-8-*-
# import argparse
# import os
# import random
# import numpy as np
# import cv2
# import glob
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from scipy.signal import savgol_filter
# import onnxruntime as ort
# from collections import OrderedDict
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision


# from tqdm import tqdm


# def infer_bs(root_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     base_options = python.BaseOptions(model_asset_path="./data_utils/blendshape_capture/face_landmarker.task")
#     options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                            output_face_blendshapes=True,
#                                            output_facial_transformation_matrixes=True,
#                                            num_faces=1)
#     detector = vision.FaceLandmarker.create_from_options(options)

#     for i in os.listdir(root_path):
#         if i.endswith(".mp4"):
#             mp4_path = os.path.join(root_path, i)
#             npy_path = mp4_path.replace('.mp4', '_bs.npy') #os.path.join(root_path, "bs.npy")
#             if os.path.exists(npy_path):
#                 print("npy file exists:", i.split(".")[0])
#                 continue
#             else:
#                 print("npy file not exists:", i.split(".")[0])
#                 image_path = os.path.join(root_path, "img/temp.png")
#                 os.makedirs(os.path.join(root_path, 'img/'), exist_ok=True)
#                 cap = cv2.VideoCapture(mp4_path)
#                 fps = cap.get(cv2.CAP_PROP_FPS)
#                 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 print("fps:", fps)
#                 print("frame_count:", frame_count)
#                 k = 0
#                 total = frame_count
#                 bs = np.zeros((int(total), 52), dtype=np.float32)
#                 print("total:", total)
#                 print("videoPath:{} fps:{},k".format(mp4_path.split('/')[-1], fps))
#                 pbar = tqdm(total=int(total))
#                 while (cap.isOpened()):
#                     ret, frame = cap.read()
#                     if ret:
#                         cv2.imwrite(image_path, frame)
#                         image = mp.Image.create_from_file(image_path)
#                         result = detector.detect(image)
#                         face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in
#                                                    result.face_blendshapes[0]]
#                         blendshape_coef = np.array(face_blendshapes_scores)[1:]
#                         blendshape_coef = np.append(blendshape_coef, 0)
#                         bs[k] = blendshape_coef
#                         pbar.update(1)
#                         k += 1
#                     else:
#                         break
#                 cap.release()
#                 pbar.close()
#                 # np.save(npy_path, bs)
#                 # print(np.shape(bs))
#                 output = np.zeros((bs.shape[0], bs.shape[1]))
#                 for j in range(bs.shape[1]):
#                     output[:, j] = savgol_filter(bs[:, j], 5, 3)
#                 np.save(npy_path, output)
#                 print(np.shape(output))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", type=str, help="idname of target person")
#     args = parser.parse_args()
#     infer_bs(args.path)