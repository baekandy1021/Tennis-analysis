from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import sys
sys.path.append('../')
import constants
from utils import measure_distance, get_center_of_bbox, get_foot_position

class PlayerTracker:
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

    # Real-world positions (meters) of the 14 court keypoints, same order as the
    # keypoint model / mini court: x across the doubles width, y from the far baseline.
    _W, _L = constants.DOUBLE_LINE_WIDTH, constants.HALF_COURT_LINE_HEIGHT * 2
    _A, _S, _N = constants.DOUBLE_ALLY_DIFFERENCE, constants.SINGLE_LINE_WIDTH, constants.NO_MANS_LAND_HEIGHT
    COURT_KEYPOINTS_METERS = np.array([
        (0, 0), (_W, 0), (0, _L), (_W, _L),                    # doubles corners
        (_A, 0), (_A, _L), (_A + _S, 0), (_A + _S, _L),        # singles corners
        (_A, _N), (_A + _S, _N), (_A, _L - _N), (_A + _S, _L - _N),  # service line ends
        (_W / 2, _N), (_W / 2, _L - _N),                       # center service line ends
    ], dtype=np.float32)

    def __init__(self, model_path, side_margin_m=1.5, max_behind_baseline_m=6.0):
        self.model = YOLO(model_path)
        self.side_margin_m = side_margin_m
        self.max_behind_baseline_m = max_behind_baseline_m
        self._homography_cache = (None, None)

    def _court_homography(self, court_keypoints):
        key = tuple(np.round(np.asarray(court_keypoints, dtype=np.float32), 1))
        if self._homography_cache[0] != key:
            image_points = np.asarray(court_keypoints, dtype=np.float32).reshape(-1, 2)
            homography, _ = cv2.findHomography(image_points, self.COURT_KEYPOINTS_METERS, cv2.RANSAC)
            self._homography_cache = (key, homography)
        return self._homography_cache[1]

    def court_position_meters(self, court_keypoints, bbox):
        homography = self._court_homography(court_keypoints)
        if homography is None:
            return None
        foot = np.array([[get_foot_position(bbox)]], dtype=np.float32)
        return cv2.perspectiveTransform(foot, homography)[0][0]

    def choose_players(self, court_keypoints, player_dict):
        """Pick at most one player per side of the net, using foot positions mapped to
        court meters. People outside the doubles width (+ margin) -- umpire, line judges,
        net-side ball kids -- or too far behind a baseline (back-wall ball kids) are
        dropped; on each side the person nearest the court's center line and closest to
        the baseline area wins."""
        width, length = self._W, self._L
        best = {}  # side -> (score, track_id)
        for track_id, bbox in player_dict.items():
            pos = self.court_position_meters(court_keypoints, bbox)
            if pos is None:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not (-self.side_margin_m <= x <= width + self.side_margin_m):
                continue
            if not (-self.max_behind_baseline_m <= y <= length + self.max_behind_baseline_m):
                continue
            side = "far" if y < length / 2 else "near"
            behind = max(0.0, -y, y - length)
            score = abs(x - width / 2) + 0.3 * behind
            if side not in best or score < best[side][0]:
                best[side] = (score, track_id)
        return [track_id for _, track_id in best.values()]

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