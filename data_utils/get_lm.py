import os
import glob
import cv2
import face_alignment
import tqdm
import numpy as np
import shutil
import tempfile

def extract_landmarks(ori_imgs_dir):
    """
    Extracts facial landmarks from all .jpg images in the specified directory.
    Saves the landmarks as .lms files with the same base filename.
    """
    print(f'[INFO] ===== Extracting face landmarks from {ori_imgs_dir} =====')
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    except AttributeError:
        # Fallback for older versions of face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    for image_path in tqdm.tqdm(image_paths, desc="Processing frames"):
        input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
        if input_image is None:
            print(f'[WARNING] Unable to read image: {image_path}')
            continue
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input_image)
        if preds is not None and len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            lms_filename = os.path.splitext(image_path)[0] + '.lms'
            np.savetxt(lms_filename, lands, fmt='%.6f')
    
    del fa
    print(f'[INFO] ===== Finished extracting landmarks from {ori_imgs_dir} =====\n')

def extract_frames_from_video(video_path, frames_dir):
    """
    Extracts frames from the given video and saves them as .jpg images in frames_dir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video file: {video_path}')
        return False
    
    os.makedirs(frames_dir, exist_ok=True)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    pbar = tqdm.tqdm(total=frame_count, desc=f"Extracting frames from {os.path.basename(video_path)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    return True

def process_videos_in_folder(main_dir):
    """
    Processes all videos in each subfolder of main_dir.
    For each video, extracts frames, extracts landmarks, and cleans up temporary files.
    """
    # Supported video extensions
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')

    # Iterate through each subfolder in the main directory
    subfolders = [f.path for f in os.scandir(main_dir) if f.is_dir()]
    for subfolder in subfolders:
        print(f'[INFO] Processing subfolder: {subfolder}')
        # Find all videos in the current subfolder
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(subfolder, ext)))
        
        if not video_paths:
            print(f'[WARNING] No videos found in {subfolder}. Skipping...\n')
            continue
        
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f'[INFO] Processing video: {video_name}')
            
            # Create a temporary directory to store frames
            with tempfile.TemporaryDirectory() as temp_frames_dir:
                success = extract_frames_from_video(video_path, temp_frames_dir)
                if not success:
                    print(f'[ERROR] Failed to extract frames from {video_path}. Skipping...\n')
                    continue
                
                # Extract landmarks from the frames
                extract_landmarks(temp_frames_dir)
                
                # Optionally, move .lms files to a desired output directory
                # For example, saving .lms files alongside the video
                lms_files = glob.glob(os.path.join(temp_frames_dir, '*.lms'))
                output_lms_dir = os.path.join(subfolder, f'{video_name}_2dlandmarks')
                os.makedirs(output_lms_dir, exist_ok=True)
                
                for lms_file in lms_files:
                    shutil.move(lms_file, output_lms_dir)
                
                print(f'[INFO] Landmarks saved to {output_lms_dir}\n')
                # Temporary directory and its contents are automatically deleted

    print('[INFO] ===== All videos have been processed =====')

if __name__ == "__main__":
    import argparse
    import torch

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract facial landmarks from videos.")
    parser.add_argument('--root', type=str, help='Path to the main directory containing subfolders with videos.')
    args = parser.parse_args()

    main_dir = args.root

    if not os.path.isdir(main_dir):
        print(f'[ERROR] The specified main directory does not exist: {main_dir}')
        exit(1)
    
    process_videos_in_folder(main_dir)
