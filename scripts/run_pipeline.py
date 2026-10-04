import os
import sys
import argparse

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cv_pipeline.config import config
from cv_pipeline.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Gesto-Drive CV Pipeline")
    
    parser.add_argument("--camera", type=str, default=str(config.camera_index), help="OpenCV camera index or video file path")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (overrides --camera)")
    parser.add_argument("--width", type=int, default=config.frame_width, help="Capture width")
    parser.add_argument("--height", type=int, default=config.frame_height, help="Capture height")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=config.device, help="Compute device")
    parser.add_argument("--no-cv-window", action="store_true", help="Disable OpenCV debug window")
    parser.add_argument("--sink", type=str, choices=["stdout", "file", "both"], default=config.payload_sink, help="Payload sink")
    parser.add_argument("--output-file", type=str, default=config.payload_file, help="Path for JSONL payloads")
    parser.add_argument("--fps-target", type=float, default=config.fps_target, help="Target FPS before degradation triggers")
    parser.add_argument("--offline", action="store_true", help="Run offline without network model checks")
    parser.add_argument("--scene", type=str, choices=["outdoor", "indoor"], default=config.scene_mode, help="Set active vocabulary scene mode")
    parser.add_argument("--legacy-detector", action="store_true", help="Use legacy YOLOv8n detector instead of YOLOE")
    
    args = parser.parse_args()
    
    if args.offline:
        os.environ["TORCH_HUB_OFFLINE"] = "1"
        os.environ["ULTRALYTICS_OFFLINE"] = "1"
        config.offline = True
    
    # Input source resolution
    input_src = args.video if args.video else args.camera
    if isinstance(input_src, str) and input_src.isdigit():
        config.camera_index = int(input_src)
    else:
        config.camera_index = input_src

    config.frame_width = args.width
    config.frame_height = args.height
    config.device = args.device
    config.show_cv_window = not args.no_cv_window
    config.payload_sink = args.sink
    config.payload_file = args.output_file
    config.fps_target = args.fps_target
    config.scene_mode = args.scene
    
    # Check for legacy rollback
    if args.legacy_detector:
        config.yoloe_model = config.yolo_model
        
    try:
        pipeline = Pipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"[FATAL] Pipeline crashed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
