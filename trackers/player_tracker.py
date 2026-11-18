from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import sys
sys.path.append('../')
import constants
from utils import measure_distance, get_center_of_bbox, get_foot_position

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """Pick the 2 on-court players fresh each frame and relabel them to stable
        canonical IDs 1/2, instead of locking onto 2 of YOLO's persistent track IDs from
        frame 0. YOLO's tracker (persist=True) loses/reassigns IDs across occlusion and
        fast motion, so a fixed-ID filter can end up matching nobody for most of the video.
        Canonical IDs are assigned by nearest-position match to the previous frame's
        labeled players (falling back to depth/foot-position ordering when there's no
        previous reference), which keeps identity stable without depending on tracker IDs."""
        filtered_player_detections = []
        prev_positions = {}

        for player_dict in player_detections:
            chosen_ids = self.choose_players(court_keypoints, player_dict)
            candidates = {track_id: player_dict[track_id] for track_id in chosen_ids if track_id in player_dict}

            relabeled = {}
            remaining = dict(candidates)

            if prev_positions:
                assignments = []
                for track_id, bbox in candidates.items():
                    pos = get_center_of_bbox(bbox)
                    for label, prev_pos in prev_positions.items():
                        assignments.append((measure_distance(pos, prev_pos), track_id, label))
                assignments.sort(key=lambda x: x[0])

                used_labels, used_tracks = set(), set()
                for _, track_id, label in assignments:
                    if label in used_labels or track_id in used_tracks:
                        continue
                    relabeled[label] = candidates[track_id]
                    used_labels.add(label)
                    used_tracks.add(track_id)
                    remaining.pop(track_id, None)

            available_labels = [label for label in (1, 2) if label not in relabeled]
            for track_id, bbox in sorted(remaining.items(), key=lambda item: get_foot_position(item[1])[1]):
                if not available_labels:
                    break
                relabeled[available_labels.pop(0)] = bbox

            filtered_player_detections.append(relabeled)
            prev_positions = {label: get_center_of_bbox(bbox) for label, bbox in relabeled.items()}

        return filtered_player_detections

    def choose_players(self,court_keypoints, player_dict):
        """Pick up to 2 track IDs most likely to be the actual players, not ball boys/
        spectators. First restricts to detections whose foot position falls inside the
        court boundary (with a small margin for players stepping just outside the lines
        chasing a shot) -- ball kids stand along the fence outside this boundary. Falls
        back to the full detection set if fewer than 2 pass that filter (e.g. both players
        briefly stepped out of frame), then ranks by nearest court keypoint as before."""
        if not player_dict:
            return []

        on_court = self._filter_on_court(court_keypoints, player_dict)
        candidates = on_court if len(on_court) >= 2 else player_dict

        distances = []
        for track_id, bbox in candidates.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints),2):
                court_keypoint = (court_keypoints[i],court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance= distance
            distances.append((track_id, min_distance))

        distances.sort(key = lambda x: x[1])

        return [track_id for track_id, _ in distances[:2]]

    def _filter_on_court(self, court_keypoints, player_dict):
        court_polygon = np.array([
            (court_keypoints[0], court_keypoints[1]),  # top-left
            (court_keypoints[2], court_keypoints[3]),  # top-right
            (court_keypoints[6], court_keypoints[7]),  # bottom-right
            (court_keypoints[4], court_keypoints[5]),  # bottom-left
        ], dtype=np.float32).reshape(-1, 1, 2)

        top_width = measure_distance((court_keypoints[0], court_keypoints[1]), (court_keypoints[2], court_keypoints[3]))
        bottom_width = measure_distance((court_keypoints[4], court_keypoints[5]), (court_keypoints[6], court_keypoints[7]))
        margin_pixels = (2.0 / constants.DOUBLE_LINE_WIDTH) * ((top_width + bottom_width) / 2)

        on_court = {}
        for track_id, bbox in player_dict.items():
            foot_position = get_foot_position(bbox)
            distance_to_court = cv2.pointPolygonTest(court_polygon, foot_position, True)
            if distance_to_court >= -margin_pixels:
                on_court[track_id] = bbox

        return on_court
    


    def detect_frames(self, frames,read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        for frame in frames: 
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections,f)
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result 

        return player_dict 


    def draw_bboxes(self,video_frames,player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1,y1,x2,y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2)
                cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)), (0,0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames