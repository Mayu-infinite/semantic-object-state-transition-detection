from pathlib import Path
import csv

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_TO_ID = {
    "background": 0,
    "initial": 1,
    "transitioning": 2,
    "end": 3,
}


class RGBTransitionDataset(Dataset):
    def __init__(
        self,
        video_dir="data/custom_transitions/videos",
        annotation_file="data/custom_transitions/annotations/frames.csv",
        sequence_length=8,
        stride=4,
    ):
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        grouped = {}

        with open(annotation_file, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                video = row["video"]
                frame = int(row["frame"])
                label = row["label"].strip().lower()

                if label not in CLASS_TO_ID:
                    raise ValueError(f"Unknown label: {label}")

                grouped.setdefault(video, {})[frame] = CLASS_TO_ID[label]

        self.samples = []

        for video, annotations in grouped.items():
            frames = sorted(annotations)

            if len(frames) < sequence_length:
                continue

            for start in range(0, len(frames) - sequence_length + 1, stride):
                window = frames[start:start + sequence_length]

                # Require consecutive frame numbers.
                if window[-1] - window[0] != sequence_length - 1:
                    continue

                self.samples.append((video, window, annotations))

        if not self.samples:
            raise RuntimeError("No valid video sequences found in annotations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_name, frame_numbers, annotations = self.samples[index]
        video_path = self.video_dir / video_name

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames = []

        try:
            for frame_number in frame_numbers:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = capture.read()

                if not success:
                    raise RuntimeError(
                        f"Could not read frame {frame_number} from {video_name}"
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                frames.append(self.transform(image))
        finally:
            capture.release()

        video_tensor = torch.stack(frames)
        labels = torch.tensor(
            [annotations[n] for n in frame_numbers],
            dtype=torch.long,
        )

        return video_tensor, labels
