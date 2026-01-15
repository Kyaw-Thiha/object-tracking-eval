import cv2
import sys
import os

def concat_videos_side_by_side(video1_path, video2_path, output_path):
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one of the videos.")
        return

    # Get video properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Sanity checks
    if frame_count1 != frame_count2:
        print("Warning: Videos have different number of frames!")
    if height1 != height2:
        print("Error: Videos must have the same height.")
        return
    if abs(fps1 - fps2) > 0.1:
        print("Warning: Videos have different FPS values, using first video's FPS.")

    # Define the codec and output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_width = width1 + width2
    out_height = height1
    out = cv2.VideoWriter(output_path, fourcc, fps1, (out_width, out_height))

    # Process frame by frame
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Concatenate side by side
        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    print(f"Side-by-side video saved to: {output_path}")





if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python concat_videos.py <video1.mp4> <video2.mp4> <output.mp4>")
    #     sys.exit(1)

    # video1 = sys.argv[1]
    # video2 = sys.argv[2]
    # output = sys.argv[3]

    # concat_videos_side_by_side(video1, video2, output)
    videos_type1 = "./annotated_videos_bytetrack/"
    videos_type2 = "./annotated_videos_pseudouncertain/"

    type1_content = os.listdir(videos_type1)
    type2_content = os.listdir(videos_type2)

    os.makedirs("./comparison_videos/", exist_ok=True)

    for video_name in type1_content:
        if video_name in type2_content:
            print(f"Processing {video_name}...")
            video1_path = os.path.join(videos_type1, video_name)
            video2_path = os.path.join(videos_type2, video_name)
            output_path = os.path.join("./comparison_videos/", video_name)
            concat_videos_side_by_side(video1_path, video2_path, output_path)
    print("All videos processed.")

