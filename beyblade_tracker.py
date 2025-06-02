import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import pandas as pd
import os
import glob

PREDEFINED_COLORS = {
    "RED": (0, 0, 255), "DARK_RED": (0, 0, 139),
    "BLUE": (255, 0, 0), "LIGHT_BLUE": (230, 216, 173), "DARK_BLUE": (139, 0, 0),
    "GREEN": (0, 255, 0), "DARK_GREEN": (0, 100, 0),
    "YELLOW": (0, 255, 255),
    "ORANGE": (0, 165, 255), "DARK_ORANGE": (0, 140, 255),
    "PURPLE": (128, 0, 128), "MAGENTA": (255, 0, 255),
    "BLACK": (50, 50, 50), "WHITE": (240, 240, 240),
    "GREY": (128, 128, 128), "PINK": (203, 192, 255),
    "CYAN": (255, 255, 0), "BROWN": (42, 42, 165)
}

# Create necessary directories
def create_directories():
    directories = ['input', 'output', 'model']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Find model file in model directory
def find_model_file():
    model_dir = "model"
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt model file found in {model_dir} directory. Please place your model file there.")
    return model_files[0]  # Return the first .pt file found

# Find video file in input directory
def find_video_file():
    input_dir = "input"
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, extension)))
    
    if not video_files:
        raise FileNotFoundError(f"No video file found in {input_dir} directory. Please place your video file there.")
    return video_files[0]  # Return the first video file found

# Find closest color name from BGR values
def get_color_name(bgr_tuple, color_palette):
    min_dist = float('inf')
    closest_color_name = "UNKNOWN"
    bgr_np = np.array(bgr_tuple)
    for name, color_bgr in color_palette.items():
        dist = np.linalg.norm(bgr_np - np.array(color_bgr))
        if dist < min_dist:
            min_dist = dist
            closest_color_name = name
    return closest_color_name

# Get next battle number from existing CSV
def get_next_battle_number(csv_filename):
    csv_path = os.path.join("output", csv_filename)
    if not os.path.exists(csv_path):
        return 1
    df = pd.read_csv(csv_path)
    next_battle = len(df) + 1
    return next_battle

# Save battle results to CSV file
def save_battle_result(csv_filename, battle_number, duration, winner, winner_spin_duration, faster_spinner_info_val):
    csv_path = os.path.join("output", csv_filename)
    new_data = pd.DataFrame({
        'Battle_Number': [battle_number],
        'Battle_Duration': [duration],
        'Winner': [winner],
        'Winner_Spin_Duration': [winner_spin_duration],
        'Fastest_Spinner_First_Half': [faster_spinner_info_val]
    })
    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(csv_path, index=False)

class BeybladeTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = 100
        self.max_disappeared = 15

    # Get average color from center region of bounding box
    def _get_roi_avg_bgr(self, frame, bbox):
        x, y, w, h = bbox
        roi_center_x, roi_center_y = x + w // 2, y + h // 2
        roi_dim = min(w, h) // 2
        roi_x1 = max(0, roi_center_x - roi_dim // 2)
        roi_y1 = max(0, roi_center_y - roi_dim // 2)
        roi_x2 = min(frame.shape[1], roi_center_x + roi_dim // 2)
        roi_y2 = min(frame.shape[0], roi_center_y + roi_dim // 2)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        return np.mean(roi.reshape(-1, 3), axis=0)


    # Match detections to existing tracks and create new tracks
    def update(self, detections, frame):
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []

        current_centers = []
        current_avg_bgrs = []
        for det in detections:
            x, y, w, h = det[0]
            center = (x + w//2, y + h//2)
            current_centers.append(center)
            avg_bgr = self._get_roi_avg_bgr(frame, (x, y, w, h))
            current_avg_bgrs.append(avg_bgr)

        if len(self.tracks) == 0:
            for i, (center, avg_bgr) in enumerate(zip(current_centers, current_avg_bgrs)):
                color_name = get_color_name(avg_bgr, PREDEFINED_COLORS)
                self.tracks[self.next_id] = {
                    'center': center, 'avg_bgr': avg_bgr, 'color_name': color_name,
                    'bbox': detections[i][0], 'disappeared': 0
                }
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            distances = np.zeros((len(track_ids), len(current_centers)))
            for i, tid in enumerate(track_ids):
                track_center = self.tracks[tid]['center']
                track_avg_bgr = self.tracks[tid]['avg_bgr']
                for j, (current_center, current_avg_bgr) in enumerate(zip(current_centers, current_avg_bgrs)):
                    pos_dist = math.sqrt((track_center[0] - current_center[0])**2 + (track_center[1] - current_center[1])**2)
                    color_dist = np.linalg.norm(np.array(track_avg_bgr) - np.array(current_avg_bgr))
                    distances[i, j] = pos_dist + color_dist * 0.5

            used_tracks = set()
            used_detections = set()
            for _ in range(min(len(track_ids), len(current_centers))):
                if np.all(distances == float('inf')): break
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                i, j = min_idx
                if distances[i, j] < self.max_distance and i not in used_tracks and j not in used_detections:
                    tid = track_ids[i]
                    self.tracks[tid]['center'] = current_centers[j]
                    self.tracks[tid]['avg_bgr'] = current_avg_bgrs[j]
                    self.tracks[tid]['bbox'] = detections[j][0]
                    self.tracks[tid]['disappeared'] = 0
                    used_tracks.add(i)
                    used_detections.add(j)
                    distances[i, :] = float('inf')
                    distances[:, j] = float('inf')
                else:
                    distances[i, j] = float('inf')
            
            for i, tid in enumerate(track_ids):
                if i not in used_tracks: self.tracks[tid]['disappeared'] += 1
            
            for j in range(len(current_centers)):
                if j not in used_detections:
                    avg_bgr = current_avg_bgrs[j]
                    color_name = get_color_name(avg_bgr, PREDEFINED_COLORS)
                    self.tracks[self.next_id] = {
                        'center': current_centers[j], 'avg_bgr': avg_bgr, 'color_name': color_name,
                        'bbox': detections[j][0], 'disappeared': 0
                    }
                    self.next_id += 1

        to_remove = [tid for tid, track in self.tracks.items() if track['disappeared'] > self.max_disappeared]
        for tid in to_remove: del self.tracks[tid]
        return [(tid, track['bbox'], track['color_name']) for tid, track in self.tracks.items() if track['disappeared'] == 0]

# Convert frame count to MM:SS.mmm format
def format_time(frame_count, fps_val):
    if fps_val == 0: return "00:00.000"
    frame_count = max(0, frame_count) 
    seconds = frame_count / fps_val
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def main():
    # Create necessary directories
    create_directories()
    
    # File paths
    CSV_FILENAME = "beyblade_battles.csv"
    
    try:
        model_path = find_model_file()
        video_path = find_video_file()
        print(f"Using model: {model_path}")
        print(f"Processing video: {video_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    battle_number = get_next_battle_number(CSV_FILENAME)
    
    model = YOLO(model_path)
    tracker = BeybladeTracker()
    cap = cv2.VideoCapture(video_path)
    
    # Resize scale for video processing
    RESIZE_SCALE = 0.6
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    # Output video path
    output_video_path = os.path.join("output", "beyblade_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    CONFIDENCE_THRESHOLD = 0.5
    SPIN_SCORE_MIN = 80
    SPIN_SCORE_MAX = 123
    STOP_FRAMES = 5

    beyblade_motion = defaultdict(lambda: {"stopped_frames": 0, "spinning": True})
    frame_idx = 0
    winner = "-"
    winner_track_id = None
    stop_frame_index = defaultdict(lambda: None)

    timer_started = False
    timer_stopped = False
    timer_start_frame = 0
    timer_end_frame = 0
    data_saved = False

    winner_determined = False
    winner_determination_frame = 0
    winner_spin_end_frame = None 
    winner_spin_duration = "00:00.000" 

    beyblade_spin_scores = defaultdict(list) 
    beyblade_colors = {} 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD or model.names[cls_id] in ["1", "2", "3", "4"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

        tracks_data = tracker.update(detections, frame)

        if not timer_started and len(tracks_data) > 0:
            timer_started = True
            timer_start_frame = frame_idx

        current_time = "00:00.000"
        if timer_started and not timer_stopped:
            elapsed_frames = frame_idx - timer_start_frame
            current_time = format_time(elapsed_frames, fps)
        elif timer_stopped: 
            elapsed_frames = timer_end_frame - timer_start_frame
            current_time = format_time(elapsed_frames, fps)

        current_track_info = {tid: {"bbox": bbox, "name": name} for tid, bbox, name in tracks_data}

        for track_id, bbox, color_name in tracks_data:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            box_height = y2 - y1
            cut = int(box_height * 0.15)
            new_y1, new_y2 = y1 + cut, y2 - cut
            
            crop = gray[new_y1:new_y2, x1:x2]
            
            f = np.fft.fft2(crop)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
            spin_score = np.mean(magnitude_spectrum)

            beyblade_colors[track_id] = color_name
            if beyblade_motion[track_id]["spinning"]: 
                beyblade_spin_scores[track_id].append((frame_idx, spin_score))

            if SPIN_SCORE_MIN <= spin_score <= SPIN_SCORE_MAX:
                beyblade_motion[track_id]["spinning"] = True
                beyblade_motion[track_id]["stopped_frames"] = 0
            else:
                beyblade_motion[track_id]["stopped_frames"] += 1
                if beyblade_motion[track_id]["stopped_frames"] >= STOP_FRAMES:
                    beyblade_motion[track_id]["spinning"] = False
            
            status = "SPIN" if beyblade_motion[track_id]["spinning"] else "STOP"
            label_color = (0, 255, 0) if status == "SPIN" else (0, 0, 255)
            label = f"{color_name} - {status} ({spin_score:.1f})"
            cv2.rectangle(orig_frame, (x1, new_y1), (x2, new_y2), label_color, 2)
            cv2.putText(orig_frame, label, (x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        for tid, motion_data in beyblade_motion.items():
            if not motion_data["spinning"] and stop_frame_index[tid] is None:
                if tid in current_track_info: stop_frame_index[tid] = frame_idx

        tracked_ids_ever = set(beyblade_motion.keys())
        if len(tracked_ids_ever) >= 2 and winner == "-": 
            spinning_tids = [tid for tid in tracked_ids_ever if beyblade_motion.get(tid, {}).get("spinning", False)]
            stopped_tids = [tid for tid in tracked_ids_ever if not beyblade_motion.get(tid, {}).get("spinning", True) and stop_frame_index.get(tid) is not None]

            if len(spinning_tids) == 1 and len(stopped_tids) >= (len(tracked_ids_ever) -1) and len(tracked_ids_ever) >1 : 
                potential_winner_tid = spinning_tids[0]
                if potential_winner_tid in current_track_info : 
                    winner = current_track_info[potential_winner_tid]["name"]
                    winner_track_id = potential_winner_tid
                    winner_determined = True
                    winner_determination_frame = frame_idx
                    if timer_started and not timer_stopped:
                        timer_stopped = True
                        timer_end_frame = frame_idx
                elif potential_winner_tid in tracker.tracks: 
                     winner = tracker.tracks[potential_winner_tid]['color_name']
                     winner_track_id = potential_winner_tid
                     winner_determined = True
                     winner_determination_frame = frame_idx
                     if timer_started and not timer_stopped:
                        timer_stopped = True
                        timer_end_frame = frame_idx

            elif len(spinning_tids) == 0 and len(stopped_tids) >=1 and len(tracked_ids_ever) >0 : 
                valid_stopped_tids = [tid for tid in stopped_tids if stop_frame_index[tid] is not None]
                if valid_stopped_tids:
                    last_to_stop_tid = max(valid_stopped_tids, key=lambda tid: stop_frame_index[tid])
                    final_winner_name = "UNKNOWN"
                    if last_to_stop_tid in current_track_info:
                        final_winner_name = current_track_info[last_to_stop_tid]["name"]
                    elif last_to_stop_tid in tracker.tracks: 
                        final_winner_name = tracker.tracks[last_to_stop_tid]['color_name']
                    
                    if final_winner_name != "UNKNOWN":
                        winner = final_winner_name
                        winner_track_id = last_to_stop_tid
                        winner_determined = True
                        winner_determination_frame = stop_frame_index[last_to_stop_tid] 
                        if timer_started and not timer_stopped:
                            timer_stopped = True
                            timer_end_frame = stop_frame_index[last_to_stop_tid] 
        
        if winner_determined and winner_track_id is not None:
            current_potential_spin_frames = frame_idx - winner_determination_frame
            if winner_spin_end_frame is None: 
                winner_is_still_spinning_and_tracked = False
                if winner_track_id in current_track_info and \
                   beyblade_motion.get(winner_track_id, {}).get("spinning", False):
                    winner_is_still_spinning_and_tracked = True
                
                if winner_is_still_spinning_and_tracked:
                    winner_spin_duration = format_time(current_potential_spin_frames, fps)
                else:
                    winner_spin_end_frame = frame_idx 
                    final_spin_frames = winner_spin_end_frame - winner_determination_frame
                    winner_spin_duration = format_time(final_spin_frames, fps)
            else:
                final_spin_frames = winner_spin_end_frame - winner_determination_frame
                winner_spin_duration = format_time(final_spin_frames, fps)

        cv2.putText(orig_frame, f"BATTLE TIME: {current_time}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(orig_frame, f"WINNER: {winner}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if winner_determined:
            cv2.putText(orig_frame, f"WINNER SPIN TIME: {winner_spin_duration}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        out.write(orig_frame)
        cv2.imshow("Beyblade Detection", orig_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        frame_idx += 1

    if not data_saved:
        final_battle_duration_str = current_time 
        if timer_started:
            actual_battle_end_frame = timer_end_frame if (timer_stopped and winner != "-") else frame_idx
            battle_duration_frames = actual_battle_end_frame - timer_start_frame
            final_battle_duration_str = format_time(battle_duration_frames, fps)

        if winner_determined and winner_track_id is not None and winner_spin_end_frame is None: 
            spin_duration_frames_final = (frame_idx -1) - winner_determination_frame 
            winner_spin_duration = format_time(spin_duration_frames_final, fps)
        
        
        avg_spin_scores_map = {}
        effective_battle_end_frame_for_avg = (timer_end_frame if (timer_stopped and winner != "-") else frame_idx)
        total_battle_frames = effective_battle_end_frame_for_avg - timer_start_frame
        
        
        fifty_percent_frame = timer_start_frame + (total_battle_frames // 2)

        for tid, score_frame_tuples in beyblade_spin_scores.items():
            
            relevant_scores = [s_tuple[1] for s_tuple in score_frame_tuples if s_tuple[0] <= fifty_percent_frame]
            if relevant_scores:
                avg_spin_scores_map[tid] = np.mean(relevant_scores)

        faster_spinner_info = "N/A"
        if avg_spin_scores_map:
            if len(avg_spin_scores_map) > 1:
                min_avg_score = float('inf')
                fastest_tid = None
                for tid_key, avg_score_val in avg_spin_scores_map.items():
                    if avg_score_val < min_avg_score:
                        min_avg_score = avg_score_val
                        fastest_tid = tid_key
                
                if fastest_tid is not None and fastest_tid in beyblade_colors:
                    fastest_color_name = beyblade_colors[fastest_tid]
                    faster_spinner_info = f"{fastest_color_name} is spinning faster (first half)"
                else:
                    faster_spinner_info = "Could not determine fastest spinner (first half)"
            elif len(avg_spin_scores_map) == 1:
                single_tid = list(avg_spin_scores_map.keys())[0]
                single_color = beyblade_colors.get(single_tid, "Unknown")
                faster_spinner_info = f"Only one Beyblade ({single_color}) tracked in first half"
        else:
            faster_spinner_info = "No spin data for first half analysis"

        if winner != "-": 
            save_battle_result(CSV_FILENAME, battle_number, final_battle_duration_str, winner, winner_spin_duration, faster_spinner_info)
            data_saved = True
        elif timer_started : 
            save_battle_result(CSV_FILENAME, battle_number, final_battle_duration_str, "NO_WINNER_DETERMINED", "00:00.000", faster_spinner_info)
            data_saved = True

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()