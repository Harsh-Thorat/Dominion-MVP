"""
Video capture and buffering system
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
import threading
import time


class VideoBuffer:
    """
    Thread-safe rolling buffer for video frames
    """
    
    def __init__(self, max_size: int = 90):
        """
        Args:
            max_size: Maximum number of frames to store (90 = 3 seconds at 30 FPS)
        """
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.last_capture_time = 0
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(frame.copy())
            self.last_capture_time = time.time()
    
    def get_snapshot(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Get the most recent N frames
        
        Args:
            num_frames: Number of frames to retrieve
        
        Returns:
            List of frames (most recent first)
        """
        with self.lock:
            if len(self.buffer) < num_frames:
                return list(self.buffer)
            return list(self.buffer)[-num_frames:]
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.buffer) == 0
    
    def get_size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


class WebcamCapture:
    """
    Manages webcam capture in a separate thread
    """
    
    def __init__(self, camera_id: int = 0, fps: int = 30):
        """
        Args:
            camera_id: Camera device ID
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.fps = fps
        self.buffer = VideoBuffer()
        self.capture = None
        self.is_running = False
        self.thread = None
    
    def start(self) -> bool:
        """
        Start capturing from webcam
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            return True
        
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            return False
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        return True
    
    def stop(self):
        """Stop capturing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        self.buffer.clear()
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        while self.is_running:
            ret, frame = self.capture.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.buffer.add_frame(frame)
            else:
                time.sleep(0.01)
    
    def get_snapshot(self, num_frames: int = 30) -> List[np.ndarray]:
        """
        Get recent frames from buffer
        
        Args:
            num_frames: Number of frames to retrieve
        
        Returns:
            List of frames
        """
        return self.buffer.get_snapshot(num_frames)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent single frame"""
        snapshot = self.buffer.get_snapshot(1)
        return snapshot[0] if snapshot else None
    
    def is_active(self) -> bool:
        """Check if capture is active and receiving frames"""
        return self.is_running and not self.buffer.is_empty()


class VideoFileCapture:
    """
    Process pre-recorded video files
    """
    
    def __init__(self, video_path: str):
        """
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
    
    def get_next_chunk(
        self,
        chunk_size: int = 30,
        overlap: int = 15
    ) -> Tuple[Optional[List[np.ndarray]], float]:
        """
        Get next chunk of frames with overlap
        
        Args:
            chunk_size: Number of frames per chunk
            overlap: Number of overlapping frames
        
        Returns:
            Tuple of (frames, timestamp) or (None, 0) if video is done
        """
        if self.current_frame >= self.total_frames:
            return None, 0.0
        
        frames = []
        start_frame = max(0, self.current_frame - overlap)
        
        # Seek to start position
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        for _ in range(chunk_size):
            ret, frame = self.capture.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        timestamp = self.current_frame / self.fps
        self.current_frame += chunk_size
        
        return frames if frames else None, timestamp
    
    def get_all_chunks(
        self,
        chunk_size: int = 30,
        overlap: int = 15
    ) -> List[Tuple[List[np.ndarray], float]]:
        """
        Get all video chunks
        
        Returns:
            List of (frames, timestamp) tuples
        """
        self.current_frame = 0
        chunks = []
        
        while True:
            frames, timestamp = self.get_next_chunk(chunk_size, overlap)
            if frames is None:
                break
            chunks.append((frames, timestamp))
        
        return chunks
    
    def release(self):
        """Release video capture"""
        if self.capture:
            self.capture.release()
    
    def __del__(self):
        self.release()
