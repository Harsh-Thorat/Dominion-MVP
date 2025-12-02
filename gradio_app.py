"""
Gradio web interface for exercise form analysis
"""
import gradio as gr
import numpy as np
import time
from typing import Optional, Tuple
import threading

from video_capture import WebcamCapture, VideoFileCapture
from analysis_pipeline import ExerciseAnalysisPipeline


class ExerciseAnalysisApp:
    """
    Main application with Gradio interface
    """
    
    def __init__(self):
        self.pipeline = None
        self.webcam_capture = None
        self.is_analyzing = False
        self.current_exercise = "squat"
        self.analysis_thread = None
        self.latest_feedback = None
        self.lock = threading.Lock()
        
    def initialize_pipeline(self, model_name: str) -> str:
        """Initialize the analysis pipeline"""
        try:
            self.pipeline = ExerciseAnalysisPipeline(
                model_name=model_name,
                use_memory=True,
                enable_streaming=False
            )
            return "✅ Pipeline initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing pipeline: {e}"
    
    def start_webcam_analysis(
        self,
        exercise_type: str,
        camera_id: int = 0
    ) -> Tuple[str, str]:
        """Start real-time webcam analysis"""
        if self.pipeline is None:
            return "❌ Please initialize the pipeline first", ""
        
        if self.is_analyzing:
            return "⚠️ Analysis already running", ""
        
        try:
            # Start webcam capture
            self.webcam_capture = WebcamCapture(camera_id=camera_id)
            if not self.webcam_capture.start():
                return "❌ Could not open webcam", ""
            
            # Wait for buffer to fill
            time.sleep(1.0)
            
            # Start analysis loop
            self.current_exercise = exercise_type
            self.is_analyzing = True
            self.analysis_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True
            )
            self.analysis_thread.start()
            
            return f"✅ Started analyzing {exercise_type}", ""
            
        except Exception as e:
            return f"❌ Error: {e}", ""
    
    def stop_webcam_analysis(self) -> str:
        """Stop webcam analysis"""
        self.is_analyzing = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=3.0)
        
        if self.webcam_capture:
            self.webcam_capture.stop()
            self.webcam_capture = None
        
        return "⏹️ Analysis stopped"
    
    def _analysis_loop(self):
        """Background analysis loop for webcam"""
        last_analysis_time = 0
        analysis_interval = 0.5  # Analyze every 500ms (2 FPS)
        
        while self.is_analyzing:
            try:
                current_time = time.time()
                
                # Check if it's time for next analysis
                if current_time - last_analysis_time < analysis_interval:
                    time.sleep(0.1)
                    continue
                
                # Get frames from buffer
                frames = self.webcam_capture.get_snapshot(num_frames=30)
                
                if len(frames) < 10:  # Need minimum frames
                    time.sleep(0.1)
                    continue
                
                # Analyze frames
                feedback = self.pipeline.analyze_frames(
                    frames=frames,
                    exercise_type=self.current_exercise,
                    timestamp=current_time,
                    use_quick_mode=True
                )
                
                # Update latest feedback
                with self.lock:
                    self.latest_feedback = feedback
                
                last_analysis_time = current_time
                
            except Exception as e:
                print(f"Analysis error: {e}")
                time.sleep(0.5)
    
    def get_latest_feedback(self) -> Tuple[str, str, str]:
        """Get the latest feedback for display"""
        with self.lock:
            if self.latest_feedback is None:
                return "⏳ Analyzing...", "", ""
            
            feedback = self.latest_feedback
            
            # Format rating with color
            rating = feedback["rating"]
            if rating == "Good":
                rating_display = "🟢 " + rating
            elif rating == "Minor Issue":
                rating_display = "🟡 " + rating
            elif rating == "Major Issue":
                rating_display = "🔴 " + rating
            else:
                rating_display = "⚪ " + rating
            
            # Format feedback text
            feedback_text = feedback["feedback"]
            
            # Format metadata
            body_parts = ", ".join(feedback.get("body_parts", []))
            inference_time = feedback.get("inference_time", 0)
            metadata = f"Body parts: {body_parts if body_parts else 'None'} | Inference: {inference_time:.2f}s"
            
            return rating_display, feedback_text, metadata
    
    def analyze_uploaded_video(
        self,
        video_file,
        exercise_type: str
    ) -> Tuple[str, str]:
        """Analyze uploaded video file"""
        if self.pipeline is None:
            return "❌ Please initialize the pipeline first", ""
        
        if video_file is None:
            return "⚠️ Please upload a video file", ""
        
        try:
            # Analyze video
            all_feedback = self.pipeline.analyze_video_file(
                video_path=video_file,
                exercise_type=exercise_type
            )
            
            # Generate summary
            summary = self._format_video_summary(all_feedback)
            
            return "✅ Video analysis complete", summary
            
        except Exception as e:
            return f"❌ Error: {e}", ""
    
    def _format_video_summary(self, all_feedback: list) -> str:
        """Format video analysis summary"""
        if not all_feedback:
            return "No feedback generated"
        
        summary = f"### Analysis Summary ({len(all_feedback)} segments)\n\n"
        
        # Count ratings
        ratings = [f["rating"] for f in all_feedback]
        good = ratings.count("Good")
        minor = ratings.count("Minor Issue")
        major = ratings.count("Major Issue")
        
        summary += f"**Overall Performance:**\n"
        summary += f"- 🟢 Good: {good}/{len(ratings)}\n"
        summary += f"- 🟡 Minor Issues: {minor}/{len(ratings)}\n"
        summary += f"- 🔴 Major Issues: {major}/{len(ratings)}\n\n"
        
        # Show feedback timeline
        summary += "**Timeline:**\n\n"
        for i, feedback in enumerate(all_feedback[:10]):  # Show first 10
            timestamp = feedback["timestamp"]
            rating = feedback["rating"]
            text = feedback["feedback"]
            
            emoji = "🟢" if rating == "Good" else "🟡" if rating == "Minor Issue" else "🔴"
            summary += f"{emoji} **{timestamp:.1f}s**: {text}\n\n"
        
        if len(all_feedback) > 10:
            summary += f"*...and {len(all_feedback) - 10} more segments*\n"
        
        return summary
    
    def get_session_summary(self) -> str:
        """Get summary of current session"""
        if self.pipeline is None:
            return "No pipeline initialized"
        
        summary = self.pipeline.get_feedback_summary()
        
        if "message" in summary:
            return summary["message"]
        
        text = "### Session Summary\n\n"
        text += f"**Total Analyses:** {summary['total_analyses']}\n\n"
        text += f"**Performance Breakdown:**\n"
        text += f"- 🟢 Good: {summary['good_count']}\n"
        text += f"- 🟡 Minor Issues: {summary['minor_count']}\n"
        text += f"- 🔴 Major Issues: {summary['major_count']}\n\n"
        
        if summary['common_issues']:
            text += f"**Common Areas of Focus:** {', '.join(summary['common_issues'])}\n\n"
        
        text += f"**Average Processing Time:** {summary['avg_inference_time']:.2f}s"
        
        return text
    
    def clear_session(self) -> str:
        """Clear session history"""
        if self.pipeline:
            self.pipeline.clear_history()
            return "✅ Session cleared"
        return "⚠️ No active session"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Exercise Form Analyzer") as demo:
            gr.Markdown("# 🏋️ Exercise Form Analyzer")
            gr.Markdown("Real-time exercise form analysis using Qwen2.5-VL")
            
            with gr.Tab("Setup"):
                gr.Markdown("## 1. Initialize Pipeline")
                model_dropdown = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2-VL-7B-Instruct",
                        "Qwen/Qwen2-VL-2B-Instruct"
                    ],
                    value="Qwen/Qwen2-VL-7B-Instruct",
                    label="Model"
                )
                init_btn = gr.Button("Initialize Pipeline", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)
                
                init_btn.click(
                    fn=self.initialize_pipeline,
                    inputs=[model_dropdown],
                    outputs=[init_status]
                )
            
            with gr.Tab("Live Analysis"):
                gr.Markdown("## Webcam Analysis")
                
                with gr.Row():
                    exercise_dropdown = gr.Dropdown(
                        choices=["squat", "push-up", "pull-up", "plank", "lunge", "general"],
                        value="squat",
                        label="Exercise Type"
                    )
                    camera_id = gr.Number(value=0, label="Camera ID", precision=0)
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Analysis", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Analysis", variant="stop")
                
                webcam_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### Current Feedback")
                rating_display = gr.Textbox(label="Form Rating", interactive=False)
                feedback_display = gr.Textbox(label="Feedback", interactive=False, lines=3)
                metadata_display = gr.Textbox(label="Details", interactive=False)
                
                # Auto-refresh feedback display
                def refresh_feedback():
                    return self.get_latest_feedback()
                
                refresh_timer = gr.Timer(value=0.5)  # Update every 500ms
                refresh_timer.tick(
                    fn=refresh_feedback,
                    outputs=[rating_display, feedback_display, metadata_display]
                )
                
                start_btn.click(
                    fn=self.start_webcam_analysis,
                    inputs=[exercise_dropdown, camera_id],
                    outputs=[webcam_status, feedback_display]
                )
                
                stop_btn.click(
                    fn=self.stop_webcam_analysis,
                    outputs=[webcam_status]
                )
            
            with gr.Tab("Video Upload"):
                gr.Markdown("## Analyze Uploaded Video")
                
                video_input = gr.Video(label="Upload Video")
                video_exercise = gr.Dropdown(
                    choices=["squat", "push-up", "pull-up", "plank", "lunge", "general"],
                    value="squat",
                    label="Exercise Type"
                )
                
                analyze_video_btn = gr.Button("Analyze Video", variant="primary")
                video_status = gr.Textbox(label="Status", interactive=False)
                video_summary = gr.Markdown(label="Analysis Results")
                
                analyze_video_btn.click(
                    fn=self.analyze_uploaded_video,
                    inputs=[video_input, video_exercise],
                    outputs=[video_status, video_summary]
                )
            
            with gr.Tab("Session Stats"):
                gr.Markdown("## Session Statistics")
                
                refresh_stats_btn = gr.Button("Refresh Stats")
                stats_display = gr.Markdown()
                clear_btn = gr.Button("Clear Session", variant="stop")
                clear_status = gr.Textbox(label="Status", interactive=False)
                
                refresh_stats_btn.click(
                    fn=self.get_session_summary,
                    outputs=[stats_display]
                )
                
                clear_btn.click(
                    fn=self.clear_session,
                    outputs=[clear_status]
                )
            
            gr.Markdown("""
            ---
            ### How to Use:
            1. **Setup**: Initialize the pipeline with your chosen model
            2. **Live Analysis**: Start webcam analysis to get real-time feedback
            3. **Video Upload**: Upload a video file for comprehensive analysis
            4. **Session Stats**: View your progress and common areas for improvement
            
            **Note**: First analysis may take 10-30 seconds as the model loads. Subsequent analyses are faster (~1.5s).
            """)
        
        return demo


def main():
    """Main entry point"""
    app = ExerciseAnalysisApp()
    demo = app.create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
