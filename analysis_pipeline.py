"""
Main analysis pipeline orchestrating all components
"""
from typing import List, Dict, Any, Optional
import numpy as np
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

from video_frame_loader import VideoFrameLoader
from qwen_vlm_wrapper import Qwen2VLWrapper
from output_parser import ExerciseFeedbackParser
from prompt_templates import create_analysis_prompt, create_quick_analysis_prompt
import time


class ExerciseAnalysisPipeline:
    """
    Main pipeline for exercise form analysis
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_memory: bool = False,
        enable_streaming: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            use_memory: Whether to maintain conversation history
            enable_streaming: Whether to stream token generation
        """
        print("Initializing Exercise Analysis Pipeline...")
        
        # Initialize components
        self.frame_loader = VideoFrameLoader(
            target_frames=8,
            target_size=(448, 448),
            normalize=True
        )
        
        print("Loading vision-language model...")
        self.vlm = Qwen2VLWrapper(model_name=model_name)
        
        self.parser = ExerciseFeedbackParser()
        
        # Optional: conversation memory
        self.memory = None
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Optional: streaming callbacks
        self.callbacks = []
        if enable_streaming:
            self.callbacks.append(StreamingStdOutCallbackHandler())
        
        # Feedback history
        self.feedback_history = []
        
        print("Pipeline initialized successfully!")
    
    def analyze_frames(
        self,
        frames: List[np.ndarray],
        exercise_type: str = "general",
        timestamp: float = 0.0,
        use_quick_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a sequence of frames
        
        Args:
            frames: List of video frames (numpy arrays)
            exercise_type: Type of exercise being performed
            timestamp: Timestamp in video (seconds)
            use_quick_mode: Use simplified prompt for faster inference
        
        Returns:
            Structured feedback dictionary
        """
        start_time = time.time()
        
        try:
            # Stage 1: Preprocess frames
            doc = self.frame_loader.load_from_buffer(
                frames,
                metadata={
                    "exercise_type": exercise_type,
                    "timestamp": timestamp
                }
            )
            
            pil_frames = doc.metadata["frames"]
            
            # Stage 2: Create prompt
            if use_quick_mode:
                prompt_text = create_quick_analysis_prompt()
            else:
                prompt_template = create_analysis_prompt(exercise_type)
                prompt_text = prompt_template.format()
            
            # Add memory context if available
            if self.memory:
                chat_history = self.memory.load_memory_variables({})
                if chat_history.get("chat_history"):
                    history_context = "\n\nPrevious feedback context:\n"
                    for msg in chat_history["chat_history"][-3:]:  # Last 3 exchanges
                        history_context += f"{msg.content}\n"
                    prompt_text = history_context + "\n" + prompt_text
            
            # Stage 3: Run inference
            callbacks = self.callbacks if self.callbacks else None
            
            raw_output = self.vlm.call_with_frames(
                prompt=prompt_text,
                frames=pil_frames,
                run_manager=None  # Callbacks would be set up here for streaming
            )
            
            # Stage 4: Parse output
            parsed_feedback = self.parser.parse(raw_output)
            
            # Add metadata
            parsed_feedback["exercise_type"] = exercise_type
            parsed_feedback["timestamp"] = timestamp
            parsed_feedback["num_frames"] = len(frames)
            parsed_feedback["inference_time"] = time.time() - start_time
            
            # Save to memory if enabled
            if self.memory:
                self.memory.save_context(
                    {"input": f"Analyzing {exercise_type}"},
                    {"output": parsed_feedback["feedback"]}
                )
            
            # Add to history
            self.feedback_history.append(parsed_feedback)
            
            return parsed_feedback
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {
                "rating": "Error",
                "feedback": "Unable to analyze form. Please ensure your full body is visible.",
                "body_parts": [],
                "exercise_type": exercise_type,
                "timestamp": timestamp,
                "error": str(e),
                "inference_time": time.time() - start_time
            }
    
    def analyze_video_file(
        self,
        video_path: str,
        exercise_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Analyze entire video file
        
        Args:
            video_path: Path to video file
            exercise_type: Type of exercise
        
        Returns:
            List of feedback for each chunk
        """
        from video_capture import VideoFileCapture
        
        print(f"Processing video file: {video_path}")
        video_capture = VideoFileCapture(video_path)
        
        all_feedback = []
        chunks = video_capture.get_all_chunks(chunk_size=30, overlap=15)
        
        print(f"Found {len(chunks)} chunks to process")
        
        for i, (frames, timestamp) in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} at {timestamp:.2f}s...")
            
            feedback = self.analyze_frames(
                frames=frames,
                exercise_type=exercise_type,
                timestamp=timestamp,
                use_quick_mode=True  # Use quick mode for batch processing
            )
            
            all_feedback.append(feedback)
            print(f"  → {feedback['rating']}: {feedback['feedback']}")
        
        video_capture.release()
        return all_feedback
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary of all feedback in session
        
        Returns:
            Summary statistics
        """
        if not self.feedback_history:
            return {"message": "No feedback history available"}
        
        total = len(self.feedback_history)
        ratings = [f["rating"] for f in self.feedback_history]
        
        return {
            "total_analyses": total,
            "good_count": ratings.count("Good"),
            "minor_count": ratings.count("Minor Issue"),
            "major_count": ratings.count("Major Issue"),
            "common_issues": self._get_common_issues(),
            "avg_inference_time": sum(f.get("inference_time", 0) for f in self.feedback_history) / total
        }
    
    def _get_common_issues(self) -> List[str]:
        """Extract most common body parts mentioned"""
        from collections import Counter
        
        all_parts = []
        for feedback in self.feedback_history:
            all_parts.extend(feedback.get("body_parts", []))
        
        if not all_parts:
            return []
        
        counter = Counter(all_parts)
        return [part for part, _ in counter.most_common(3)]
    
    def clear_history(self):
        """Clear feedback history and memory"""
        self.feedback_history = []
        if self.memory:
            self.memory.clear()
    
    def reset(self):
        """Reset pipeline state"""
        self.clear_history()
        print("Pipeline reset complete")
