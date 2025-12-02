"""
Custom LangChain Document Loader for video frames
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class VideoFrameLoader(BaseLoader):
    """
    Custom loader that processes video frames for Qwen2.5-VL
    """
    
    def __init__(
        self,
        target_frames: int = 8,
        target_size: tuple = (448, 448),
        normalize: bool = True
    ):
        """
        Args:
            target_frames: Number of frames to sample from video clip
            target_size: Target resolution for each frame
            normalize: Whether to apply ImageNet normalization
        """
        self.target_frames = target_frames
        self.target_size = target_size
        self.normalize = normalize
        
        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_from_buffer(
        self,
        frame_buffer: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Load and process frames from a buffer
        
        Args:
            frame_buffer: List of raw frames (numpy arrays)
            metadata: Additional metadata about the video clip
        
        Returns:
            Document containing processed frames and metadata
        """
        if not frame_buffer:
            raise ValueError("Frame buffer is empty")
        
        # Sample frames uniformly
        sampled_frames = self._sample_frames(frame_buffer)
        
        # Preprocess each frame
        processed_frames = [
            self._preprocess_frame(frame) for frame in sampled_frames
        ]
        
        # Create metadata
        doc_metadata = {
            "num_original_frames": len(frame_buffer),
            "num_sampled_frames": len(processed_frames),
            "frame_size": self.target_size,
            **(metadata or {})
        }
        
        # Store frames as PIL Images for compatibility with Qwen
        pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
        
        return Document(
            page_content="",  # Video has no text content
            metadata={
                **doc_metadata,
                "frames": pil_frames,
                "processed_frames": processed_frames
            }
        )
    
    def load_from_file(self, video_path: str) -> List[Document]:
        """
        Load video from file and split into chunks
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of Documents, each containing a video chunk
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        documents = []
        frame_buffer = []
        frames_per_chunk = 30  # ~1 second at 30 FPS
        overlap = 15  # 50% overlap
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame)
            frame_idx += 1
            
            # Process chunk when buffer is full
            if len(frame_buffer) >= frames_per_chunk:
                metadata = {
                    "source": video_path,
                    "fps": fps,
                    "chunk_start_frame": frame_idx - frames_per_chunk,
                    "chunk_end_frame": frame_idx,
                    "timestamp": (frame_idx - frames_per_chunk) / fps
                }
                doc = self.load_from_buffer(frame_buffer, metadata)
                documents.append(doc)
                
                # Keep overlap frames
                frame_buffer = frame_buffer[-overlap:]
        
        cap.release()
        return documents
    
    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Uniformly sample target number of frames
        """
        num_frames = len(frames)
        if num_frames <= self.target_frames:
            return frames
        
        # Calculate indices for uniform sampling
        indices = np.linspace(0, num_frames - 1, self.target_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize and normalize a single frame
        """
        # Resize
        frame = cv2.resize(frame, self.target_size)
        
        # Normalize if requested
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.mean) / self.std
        
        return frame
    
    def load(self) -> List[Document]:
        """
        Required by BaseLoader interface
        """
        raise NotImplementedError(
            "Use load_from_buffer() or load_from_file() instead"
        )
