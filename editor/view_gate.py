import os
import cv2
import numpy as np
import torch
import torchvision.models as models


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class FrameEmbedder():
    """ImageNet MobileNetV3-small features (576-d). Frozen; only a linear head is trained."""

    def __init__(self):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        net = models.mobilenet_v3_small(weights=weights)
        self.backbone = torch.nn.Sequential(net.features, net.avgpool, torch.nn.Flatten()).eval().to(_device())
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, frames_bgr):
        batch = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
        x = (torch.stack(batch) - self.mean) / self.std
        return self.backbone(x.to(_device())).cpu().numpy()


class ViewClassifier():
    """Is this frame the wide main-court camera? Linear probe on FrameEmbedder features,
    trained by scripts/train_view_classifier.py from hand-sorted frames."""

    def __init__(self, model_path, threshold=0.5):
        state = torch.load(model_path, map_location="cpu")
        self.head = torch.nn.Linear(state["in_features"], 1)
        self.head.load_state_dict(state["head"])
        self.head.eval()
        self.embedder = FrameEmbedder()
        self.threshold = threshold

    @torch.no_grad()
    def court_view_prob(self, frame):
        features = torch.from_numpy(self.embedder.embed([frame]))
        return float(torch.sigmoid(self.head(features))[0, 0])

    def is_court_frame(self, frame):
        return self.court_view_prob(frame) >= self.threshold


def load_view_checker(model_path, court_model_path, reference_frame):
    """Prefer the trained classifier; fall back to the keypoint/homography check."""
    if model_path and os.path.exists(model_path):
        print(f"View gate: trained classifier ({model_path})")
        return ViewClassifier(model_path)
    from court_line_detector import CourtLineDetector
    from editor.court_keypoint_check import CourtKeypointCheck
    print("View gate: no trained classifier found, using court-keypoint fallback")
    return CourtKeypointCheck(CourtLineDetector(court_model_path), reference_frame)


def vote_shots(shots, sample_labels, majority_ratio=0.6):
    """shots: [(start, end)] sample indices; sample_labels: {sample_idx: bool} for the
    sparsely checked samples. Returns one bool per shot (majority of its checked samples)."""
    results = []
    for start, end in shots:
        votes = [v for i, v in sample_labels.items() if start <= i < end]
        results.append(bool(votes) and sum(votes) / len(votes) >= majority_ratio)
    return results


def train_view_classifier(court_dir, other_dir, out_path, epochs=300, lr=1e-2):
    exts = (".jpg", ".jpeg", ".png")
    paths, labels = [], []
    for label, folder in ((1, court_dir), (0, other_dir)):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(exts):
                paths.append(os.path.join(folder, name))
                labels.append(label)
    if len(set(labels)) < 2:
        raise ValueError("Need labeled frames in both folders")

    embedder = FrameEmbedder()
    features = np.concatenate([embedder.embed([cv2.imread(p) for p in paths[i:i + 32]])
                               for i in range(0, len(paths), 32)])
    X = torch.from_numpy(features)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(y))
    n_val = max(1, len(y) // 5)
    val_idx, train_idx = order[:n_val], order[n_val:]

    head = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-3)
    pos_weight = (y[train_idx] == 0).sum() / max(1, (y[train_idx] == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(head(X[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_pred = (torch.sigmoid(head(X[val_idx])) >= 0.5).float()
        val_acc = float((val_pred == y[val_idx]).float().mean())
    print(f"Trained on {len(train_idx)} frames, held-out accuracy {val_acc:.3f} on {len(val_idx)} frames")

    # refit on everything before saving
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(head(X), y).backward()
        optimizer.step()
    torch.save({"head": head.state_dict(), "in_features": X.shape[1]}, out_path)
    print(f"Saved {out_path}")
