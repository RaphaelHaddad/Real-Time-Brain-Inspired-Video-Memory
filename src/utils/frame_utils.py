import cv2
import numpy as np
from typing import List, Tuple
from ..core.logger import get_logger

logger = get_logger(__name__)

def extract_keyframes(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """
    Extract keyframes from a video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < num_frames:
        # If video has fewer frames than requested, extract all frames
        frame_indices = list(range(total_frames))
    else:
        # Evenly distribute frame extraction across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy())
        else:
            logger.warning(f"Could not read frame at index {idx}")
    
    cap.release()
    return frames

def resize_frame(frame: np.ndarray, max_height: int = 720) -> np.ndarray:
    """
    Resize a frame to have a maximum height while maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    if height <= max_height:
        return frame
    
    scale = max_height / height
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height))

def frames_to_base64(frames: List[np.ndarray], quality: int = 85) -> List[str]:
    """
    Convert a list of frames to base64-encoded JPEG strings
    """
    import base64
    
    base64_frames = []
    for frame in frames:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        base64_str = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(f"data:image/jpeg;base64,{base64_str}")
    
    return base64_frames