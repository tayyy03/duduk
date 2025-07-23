import streamlit as st
import os
import tempfile
import numpy as np
import math

# Try to import cv2 with error handling
try:
    import cv2
except ImportError as e:
    st.error("❌ OpenCV import failed. Please check the deployment logs.")
    st.stop()

# Try to import YOLO with error handling
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error("❌ Ultralytics import failed. Please check the deployment logs.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Pose Detection & Classification",
    page_icon="🤸‍♂️",
    layout="wide"
)

CLASS_LABELS = {
    0: "Bad",
    1: "Good"
}

COLORS = {
    0: (0, 255, 0),  # Bad → hijau
    1: (255, 0, 0),  # Good → biru
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]  

@st.cache_resource
def load_model():
    model_path = "pose2/train2/weights/best.pt"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file tidak ditemukan: {model_path}")
        st.info("Struktur folder yang diharapkan: pose2/train2/weights/best.pt")
        
        # List all files in current directory for debugging
        st.write("📁 **Files in current directory:**")
        for root, dirs, files in os.walk("."):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            st.write(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                st.write(f"{subindent}{file}")
        return None
    
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()
if model is None:
    st.stop()

st.title("🤸‍♂️ Deteksi Pose & Tracking dengan Sudut (3 Keypoints)")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    img_size = st.selectbox("Image Size", [320, 640, 1280], index=1)

source = st.radio("Pilih sumber input:", ["Webcam", "Upload Video"], horizontal=True)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    if None in (a, b, c):
        return None
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print(f"Angle calculation error: {e}")
        return None

def draw_pose_with_label(frame, keypoints_obj, label, box):
    """Draw pose keypoints with labels and angles"""
    color = COLORS.get(label, (255, 255, 255))

    try:
        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy()
    except Exception as e:
        print(f"Keypoint error: {e}")
        return frame

    # Draw keypoints
    pts = []
    for i, (x, y) in enumerate(keypoints):
        if i < len(confs) and confs[i] > 0.5:
            pt = (int(x), int(y))
            pts.append(pt)
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        else:
            pts.append(None)

    # Draw connections
    for i, j in KEYPOINT_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            if pts[i] and pts[j]:
                cv2.line(frame, pts[i], pts[j], color, 2)
    
    # Calculate and display angle
    if len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
        angle = calculate_angle(pts[0], pts[1], pts[2])
        if angle is not None:
            pos = pts[1]
            cv2.putText(frame, f"{int(angle)}°", (pos[0] + 5, pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw bounding box and label
    if box is not None:
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, CLASS_LABELS.get(label, "Unknown"), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            print(f"Box drawing error: {e}")

    return frame

def process_frame(frame):
    """Process single frame with model prediction"""
    try:
        results = model.predict(frame, imgsz=img_size, conf=conf_threshold, save=False)

        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            if boxes is not None and kpts is not None:
                for box, kp in zip(boxes, kpts):
                    label = int(box.cls.cpu().item())
                    frame = draw_pose_with_label(frame, kp, label, box)
        
        return frame
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return frame

def infer_and_display_video(video_path):
    """Process uploaded video"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Cannot open video file")
            return
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FPS", fps)
        with col2:
            st.metric("Total Frames", total_frames)
        with col3:
            st.metric("Duration", f"{duration:.1f}s")
        
        stframe = st.empty()
        progress_bar = st.progress(0)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame)
            stframe.image(processed_frame, channels="BGR", use_container_width=True)
            
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(frame_count / total_frames)

        cap.release()
        st.success("✅ Video processing completed!")
        
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")

# Main interface
if source == "Webcam":
    st.subheader("📹 Webcam Real-time Detection")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run = st.checkbox("▶️ Start Webcam")
        if st.button("⏹️ Stop"):
            run = False
    
    if run:
        with col2:
            st.info("Webcam is running... Uncheck 'Start Webcam' to stop")
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Make sure no other application is using it.")
            else:
                stframe = st.empty()
                
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break

                    processed_frame = process_frame(frame)
                    stframe.image(processed_frame, channels="BGR", use_container_width=True)

                cap.release()
        except Exception as e:
            st.error(f"Webcam error: {str(e)}")

else:  # Upload Video
    st.subheader("📁 Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.info(f"📄 **File:** {uploaded_file.name} | **Size:** {file_size_mb:.2f} MB")
        
        if st.button("🚀 Process Video"):
            with st.spinner("Processing video..."):
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                try:
                    infer_and_display_video(tfile.name)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tfile.name):
                        os.unlink(tfile.name)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🤖 Powered by YOLO v8 + Streamlit | 🔗 Repository: 
        <a href="https://github.com/tayyy03/duduk" target="_blank">github.com/tayyy03/duduk</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)
