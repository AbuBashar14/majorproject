import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream depth and color frames
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start the pipeline
pipeline.start(config)

# Function to process and display depth data as a heatmap (optional)
def show_depth_frame(depth_frame):
    # Convert the depth frame to a numpy array
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Normalize depth image to 8-bit for better visualization
    depth_image = np.uint8(depth_image * 255 / np.max(depth_image))  # Normalize to 255 for visualization
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # Apply a color map for depth visualization
    
    return depth_image

try:
    while True:
        # Wait for a new frame from the camera
        frames = pipeline.wait_for_frames()

        # Get depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Optionally, process the depth image (for visualization, etc.)
        depth_image = show_depth_frame(depth_frame)

        # Display color and depth frames in OpenCV windows
        cv2.imshow("Color Frame", color_image)
        cv2.imshow("Depth Frame", depth_image)

        # Add a break condition (stop after 10 frames for testing)
        time.sleep(0.1)  # Add a delay to allow the display to update
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to break the loop
            break

finally:
    # Stop the pipeline and clean up
    pipeline.stop()
    cv2.destroyAllWindows()
