import os
import sys
import math
from PIL import Image

def single_video_det_diff(output1_path, output2_path, example_image_path):
    # get image dimensions
    img = Image.open(example_image_path)
    image_width, image_height = img.size

    # # check if two output files correspond to the same video
    # if os.path.basename(output1_path.split("/")[-1]) != os.path.basename(output2_path.split("/")[-1]):
    #     raise ValueError("Output files do not correspond to the same video.")
    # print("Comparing outputs for video: ", os.path.basename(output1_path.split("/")[-1]))
    # read output 1 file
    # read output file
    if isinstance(output1_path, list):
        outputs1 = {}
        for output_path in output1_path:
            new_outputs = read_detection_output(output_path)
            for frame_id, dets in new_outputs.items():
                outputs1.setdefault(frame_id, []).extend(dets)
    else:
        outputs1 = read_detection_output(output1_path)
    
    # read output 2 file
    if isinstance(output2_path, list):
        outputs2 = {}
        for output_path in output2_path:
            new_outputs = read_detection_output(output_path)
            for frame_id, dets in new_outputs.items():
                outputs2.setdefault(frame_id, []).extend(dets)
    else:
        outputs2 = read_detection_output(output2_path)
    
    # compare outputs frame by frame
    frame_ids = []
    diff_per_timestamp = []
    cur_frame_id = min(outputs1.keys())
    print("Starting frame id: ", cur_frame_id)
    while cur_frame_id in outputs1 and cur_frame_id in outputs2:
        output1_bboxes = outputs1[cur_frame_id]
        output2_bboxes = outputs2[cur_frame_id]
        frame_avg_diff = shortest_distance_bbox_comparison(output1_bboxes, output2_bboxes, image_width, image_height)
        frame_ids.append(cur_frame_id)
        diff_per_timestamp.append(frame_avg_diff)
        cur_frame_id += 1
    
    return frame_ids, diff_per_timestamp


def read_detection_output(output_path):
    outputs = {}
    with open(output_path, 'r') as f1:
        for line in f1:
            line = line.strip()
            # print(line)
            if len(line.split(",")) == 10:
                frame_id, track_id, x0, y0, x1, y1, score, _, _, _ = line.split(",")
            else:
                frame_id, track_id, x0, y0, x1, y1, score, _, _ = line.split(",")
            # print(frame_id, track_id, x0, y0, x1, y1, score)
            frame_id = int(float(frame_id))
            track_id = int(float(track_id))
            x0 = float(x0)
            y0 = float(y0)
            x1 = float(x1)
            y1 = float(y1)
            score = float(score)
            if frame_id not in outputs:
                outputs[frame_id] = []
            outputs[frame_id].append((track_id, x0, y0, x1, y1, score))
    return outputs


def diff_detection_by_frame(outputs1, outputs2, image_width, image_height):
    # compare outputs frame by frame
    frame_ids = []
    diff_per_timestamp = []
    cur_frame_id = min(outputs1.keys())
    print("Starting frame id: ", cur_frame_id)
    while cur_frame_id in outputs1 and cur_frame_id in outputs2:
        output1_bboxes = outputs1[cur_frame_id]
        output2_bboxes = outputs2[cur_frame_id]
        frame_avg_diff = shortest_distance_bbox_comparison(output1_bboxes, output2_bboxes, image_width, image_height)
        frame_ids.append(cur_frame_id)
        diff_per_timestamp.append(frame_avg_diff)
        cur_frame_id += 1
    
    return frame_ids, diff_per_timestamp


def shortest_distance_bbox_comparison(bbox_list1, bbox_list2, image_width, image_height):
    # build a distance score matrix between two bbox lists
    distance_score = [[float("inf") for _ in range(len(bbox_list2))] for _ in range(len(bbox_list1))]
    for i in range(len(bbox_list1)):
        for j in range(len(bbox_list2)):
            _, x0_1, y0_1, x1_1, y1_1, score_1 = bbox_list1[i]
            _, x0_2, y0_2, x1_2, y1_2, score_2 = bbox_list2[j]
            diff_x0 = abs(x0_1 - x0_2)
            diff_x1 = abs(x1_1 - x1_2)
            diff_y0 = abs(y0_1 - y0_2)
            diff_y1 = abs(y1_1 - y1_2)
            diff_score = abs(score_1 - score_2)

            norm_diff_x = 1/2 * (diff_x0 / image_width + diff_x1 / image_width)
            norm_diff_y = 1/2 * (diff_y0 / image_height + diff_y1 / image_height)
            distance = 1/2 * (norm_diff_x + norm_diff_y)
            distance_score[i][j] = distance

    # perform greedy matching based on distance score
    list_matched_pairs_scores = []
    used_i = set()
    used_j = set()
    while True:
        min_distance = float("inf")
        min_i, min_j = None, None
        for i in range(len(bbox_list1)):
            if i in used_i: continue
            for j in range(len(bbox_list2)):
                if j in used_j: continue
                if distance_score[i][j] < min_distance:
                    min_distance = distance_score[i][j]
                    min_i, min_j = i, j
        if min_i is None:  # no valid pair left
            break
        used_i.add(min_i)
        used_j.add(min_j)
        list_matched_pairs_scores.append(min_distance)
        
    avg_distance_across_bboxes = sum(list_matched_pairs_scores) / len(list_matched_pairs_scores) if len(list_matched_pairs_scores) > 0 else 1
                
    return avg_distance_across_bboxes


def plot_diff(frame_ids, diff_per_timestamp, save_path):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(frame_ids, diff_per_timestamp, marker='o')
    plt.xlabel("Frame ID")
    plt.ylabel("Average Bounding Box Distance (Normalized by Image Dimensions)")
    plt.title("Bounding Box Difference Over Frames")
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def single_video_track_diff(output_paths, gt_paths, example_image_path, match_threshold=0.001):
    # get image dimensions
    img = Image.open(example_image_path)
    image_width, image_height = img.size

    # read output file
    if isinstance(output_paths, list):
        outputs = {}
        for output_path in output_paths:
            new_outputs = read_detection_output(output_path)
            for frame_id, dets in new_outputs.items():
                outputs.setdefault(frame_id, []).extend(dets)
    else:
        outputs = read_detection_output(output_paths)
    
    # read ground truth file
    if isinstance(gt_paths, list):
        gt_outputs = {}
        for gt_path in gt_paths:
            new_gt = read_detection_output(gt_path)
            for frame_id, dets in new_gt.items():
                gt_outputs.setdefault(frame_id, []).extend(dets)
    else:
        gt_outputs = read_detection_output(gt_paths)
    
    # group each outputs by track id
    output_tracks = {}
    for frame_id in outputs:
        for det in outputs[frame_id]:
            track_id, x0, y0, x1, y1, score = det
            if track_id not in output_tracks:
                output_tracks[track_id] = {}
            output_tracks[track_id][frame_id] = (x0, y0, x1, y1, score)
    
    gt_tracks = {}
    for frame_id in gt_outputs:
        for det in gt_outputs[frame_id]:
            track_id, x0, y0, x1, y1, score = det
            if track_id not in gt_tracks:
                gt_tracks[track_id] = {}
            gt_tracks[track_id][frame_id] = (x0, y0, x1, y1, score)

    # recorded distance and coverage for each gt track
    gt_track_distance = {}
    gt_track_coverage = {}
    gt_track_apparance_rate = {}
    
    # match closest tracks between output and ground truth
    output_track_id_used = set()
    for gt_track_id in gt_tracks.keys():
        gt_start_frame_id = min(gt_tracks[gt_track_id].keys())
        gt_end_frame_id = max(gt_tracks[gt_track_id].keys())

        distance_to_each_output_track = {}
        coverage_of_each_output_track = {}

        # find a list of output_track_ids that is the closest and covers the gt_track_id
        for output_track_id in output_tracks.keys():
            output_start_frame_id = min(output_tracks[output_track_id].keys())
            output_end_frame_id = max(output_tracks[output_track_id].keys())
            # if there are overlapping frames between two tracks
            # if overlap_interval(gt_start_frame_id, gt_end_frame_id, output_start_frame_id, output_end_frame_id) > 0:
            if len(overlap_frames(gt_tracks[gt_track_id], output_tracks[output_track_id])) > 0:
                # compute distance score between two tracks
                start_frame_id = max(gt_start_frame_id, output_start_frame_id)
                end_frame_id = min(gt_end_frame_id, output_end_frame_id)
                total_distance = 0
                count = 0
                common_frames = set(gt_tracks[gt_track_id].keys()) & set(output_tracks[output_track_id].keys())
                for frame_id in common_frames:
                    if frame_id in gt_tracks[gt_track_id] and frame_id in output_tracks[output_track_id]:
                        x0_1, y0_1, x1_1, y1_1, score_1 = gt_tracks[gt_track_id][frame_id]
                        x0_2, y0_2, x1_2, y1_2, score_2 = output_tracks[output_track_id][frame_id]
                        diff_x0 = abs(x0_1 - x0_2)
                        diff_x1 = abs(x1_1 - x1_2)
                        diff_y0 = abs(y0_1 - y0_2)
                        diff_y1 = abs(y1_1 - y1_2)

                        norm_diff_x = 1/2 * (diff_x0 / image_width + diff_x1 / image_width)
                        norm_diff_y = 1/2 * (diff_y0 / image_height + diff_y1 / image_height)
                        distance = 1/2 * (norm_diff_x + norm_diff_y)

                        total_distance += distance
                        count += 1
                if count > 0:
                    avg_distance = total_distance / count
                    distance_to_each_output_track[output_track_id] = avg_distance
                    coverage_of_each_output_track[output_track_id] = (start_frame_id, end_frame_id)
        
        # select the closest sets of output tracks to cover the gt track
        # sort output track id by distance
        sorted_distance_to_each_output_track = sorted(distance_to_each_output_track.items(), key=lambda item: item[1])
        sorted_distance_to_each_output_track = dict(sorted_distance_to_each_output_track)
        # record which sections of the gt track have been covered
        gt_track_sections_not_covered = [(gt_start_frame_id, gt_end_frame_id)]
        coverage = 0
        avg_distance = float("inf")
        for output_track_id in sorted_distance_to_each_output_track.keys():
            if not output_track_id in output_track_id_used:
                if distance_to_each_output_track[output_track_id] <= match_threshold:
                    # check how much this output track can cover the gt track
                    for section in gt_track_sections_not_covered:
                        section_start, section_end = section
                        coverage_start, coverage_end = coverage_of_each_output_track[output_track_id]

                        # build a copy of gt_track section that is not covered
                        gt_track_section = {}
                        for frame_id in range(section_start, section_end + 1):
                            if frame_id in gt_tracks[gt_track_id]:
                                gt_track_section[frame_id] = gt_tracks[gt_track_id][frame_id]
                        
                        # if section_start <= coverage_start < coverage_end <= section_end: # strict
                        # if overlap_interval(section_start, section_end, coverage_start, coverage_end) > 0:
                        if len(overlap_frames(gt_track_section, output_tracks[output_track_id])) > 0:
                            # this output track can cover part of the gt track
                            # update the uncovered sections of the gt track
                            gt_track_sections_not_covered.remove(section)
                            if section_start < coverage_start:
                                gt_track_sections_not_covered.append((section_start, coverage_start - 1))
                            if coverage_end < section_end:
                                gt_track_sections_not_covered.append((coverage_end + 1, section_end))
                            output_track_id_used.add(output_track_id)
                            
                            # update average distance
                            if avg_distance == float("inf"):
                                avg_distance = distance_to_each_output_track[output_track_id]
                            else:
                                # weighted average distance, update newly covered part + old average on already covered part
                                new_section_len = coverage_end - coverage_start + 1
                                avg_distance = (
                                    distance_to_each_output_track[output_track_id] * new_section_len
                                    + avg_distance * coverage
                                ) / (coverage + new_section_len)

                            coverage += len(overlap_frames(gt_track_section, output_tracks[output_track_id])) 

        normalized_coverage = coverage / len(gt_tracks[gt_track_id])
        gt_track_distance[gt_track_id] = avg_distance
        gt_track_coverage[gt_track_id] = normalized_coverage
        total_frames = max(gt_outputs.keys())
        gt_track_apparance_rate[gt_track_id] = len(gt_tracks[gt_track_id]) / total_frames
    
    return gt_track_distance, gt_track_coverage, gt_track_apparance_rate


def overlap_interval(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))

def overlap_frames(track1, track2):
    frames1 = set(track1.keys())
    frames2 = set(track2.keys())
    return frames1 & frames2


def plot_track_diff(gt_track_distance, gt_track_coverage, gt_track_apparance_rate, save_path):
    import matplotlib.pyplot as plt

    track_ids = list(gt_track_distance.keys())
    distances = [gt_track_distance[track_id] for track_id in track_ids]
    coverages = [gt_track_coverage[track_id] for track_id in track_ids]
    appearance_rates = [gt_track_apparance_rate[track_id] for track_id in track_ids]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.bar(track_ids, distances)
    plt.xlabel("Ground Truth Track ID")
    plt.ylabel("Average Distance to Matched Output Tracks")
    plt.title("Error between GT and Output Tracks in Distance")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.bar(track_ids, coverages)
    plt.xlabel("Ground Truth Track ID")
    plt.ylabel("Coverage by Matched Output Tracks")
    plt.title("Track Coverage")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.bar(track_ids, appearance_rates)
    plt.xlabel("Ground Truth Track ID")
    plt.ylabel("proportion of frames track appeared in video")
    plt.title("Ground Truth Track appearance rate")
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

            

if __name__ == "__main__":
    # if comparing with ground truth, make sure ground truth file is the second file. 
    output_1_paths = ['/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_mot17_half_train_uncertainty_tracker/MOT17-02-DPM.txt',
                      '/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_mot17_half_val_uncertainty_tracker/MOT17-02-DPM.txt']
    
    output_2_paths = ['/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_mot17_half_train_probabilistic_byte_tracker/MOT17-02-DPM.txt',
                      '/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_mot17_half_val_probabilistic_byte_tracker/MOT17-02-DPM.txt'

    ]

    output_3_paths = ['/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_image_noise_mot17_half_train_probabilistic_byte_tracker/MOT17-02-DPM.txt',
                      '/home/allynbao/project/UncertaintyTrack/src/outputs/testrun_image_noise_mot17_half_val_probabilistic_byte_tracker/MOT17-02-DPM.txt']
    
    det_path = "/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/det/det.txt"

    gt_paths = ["/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/gt/gt_half-train.txt",
                "/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/gt/gt_half-val.txt"]
    
    example_image_path = "/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg"
    # det_plot_save_path = "/home/allynbao/project/UncertaintyTrack/src/outputs/MOT17-02-DPM_uncertaintytracker_diff_det"
    det_plot_save_path = "/home/allynbao/project/UncertaintyTrack/src/outputs/MOT17-02-DPM_image_noise_prob_bytetracker_diff_det"

    frame_ids, diff_per_timestamp = single_video_det_diff(output_1_paths, det_path, example_image_path)
    gt_track_distance, gt_track_coverage, gt_track_apparance_rate = single_video_track_diff(output_1_paths, gt_paths, example_image_path, match_threshold=0.01)
    
    print("GT Track Distance: ", gt_track_distance)
    print("GT Track Coverage: ", gt_track_coverage)
    
    track_plot_save_path = "/home/allynbao/project/UncertaintyTrack/src/outputs/MOT17-02-DPM_image_noise_prob_bytetracker_diff_track"
    plot_diff(frame_ids, diff_per_timestamp, det_plot_save_path)
    plot_track_diff(gt_track_distance, gt_track_coverage, gt_track_apparance_rate, track_plot_save_path)