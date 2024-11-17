import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream depth and color frames
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start the pipeline
pipeline.start(config)

def show_frame(frame):
    """Display the frame using OpenCV's imshow (for VSCode)"""
    cv2.imshow('Color Frame', frame)
    cv2.waitKey(1)  # To display the image for 1ms, you can adjust this to your needs

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Ensure both frames are valid
        if not depth_frame or not color_frame:
            print("Error: Could not get a frame.")
            continue  # Skip to the next iteration if no valid frame

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Optionally process depth image (visualization, etc.)
        # For example, you can display depth as a grayscale image:
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # show_frame(depth_colormap)

        # Display the color image in an OpenCV window
        show_frame(color_image)

        # Add a break condition (e.g., break after one frame for testing)
        break  # Comment this line to run indefinitely

finally:
    # Stop the pipeline and clean up
    pipeline.stop()
    cv2.destroyAllWindows()  # Close any OpenCV windows when done
