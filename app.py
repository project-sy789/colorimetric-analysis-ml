"""
Colorimetric Analysis Application

แอปพลิเคชัน Streamlit สำหรับการวิเคราะห์ค่าความเข้มข้นของสารจากค่าสี RGB
โดยใช้ Machine Learning (Random Forest Regressor)

Modules:
    - Data Collection: รวบรวมข้อมูล RGB และค่าความเข้มข้นจากภาพ
    - Model Training: เทรนโมเดล ML จากข้อมูลที่รวบรวม
    - Prediction: ทำนายค่าความเข้มข้นจากภาพใหม่

Author: Colorimetric Analysis Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os


def get_profile_files():
    """
    Get dataset and model filenames for the current profile.
    
    Returns:
        tuple: (dataset_file, model_file) based on current profile
    """
    if 'current_profile' in st.session_state:
        profile = st.session_state.current_profile.lower().replace(' ', '_')
        return f"{profile}_dataset.csv", f"{profile}_model.joblib"
    return "dataset.csv", "model.joblib"


def validate_image_format(filename):
    """
    Validate if the uploaded file has a supported image format.
    
    Args:
        filename: Name of the uploaded file
    
    Returns:
        bool: True if format is valid
        
    Raises:
        ValueError: If format is not supported
    """
    if not filename:
        raise ValueError("ไม่พบชื่อไฟล์")
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    file_ext = os.path.splitext(filename)[1]
    
    if file_ext not in valid_extensions:
        raise ValueError(f"รูปแบบไฟล์ไม่ถูกต้อง ({file_ext}) กรุณาอัปโหลด PNG, JPG หรือ JPEG")
    
    return True


def draw_roi_on_image(image_array, roi_size=100, x_offset=0, y_offset=0):
    """
    Draw ROI rectangle on image for visualization.
    
    This function draws a rectangle on the image to show the Region of Interest
    that will be used for RGB calculation.
    
    Args:
        image_array: NumPy array of the image (RGB format)
        roi_size (int): Size of the square ROI in pixels
        x_offset (int): Horizontal offset from center in pixels
        y_offset (int): Vertical offset from center in pixels
    
    Returns:
        numpy.ndarray: Image with ROI rectangle drawn on it
    """
    # Create a copy to avoid modifying the original
    img_with_roi = image_array.copy()
    
    # Get image dimensions
    height, width = img_with_roi.shape[:2]
    
    # Calculate ROI center position
    center_y = height // 2
    center_x = width // 2
    roi_center_x = center_x + x_offset
    roi_center_y = center_y + y_offset
    
    # Calculate half of ROI size
    half_roi = roi_size // 2
    
    # Calculate ROI boundaries
    y_start = max(0, roi_center_y - half_roi)
    y_end = min(height, roi_center_y + half_roi)
    x_start = max(0, roi_center_x - half_roi)
    x_end = min(width, roi_center_x + half_roi)
    
    # Draw rectangle on image
    # Convert RGB to BGR for OpenCV drawing
    img_bgr = cv2.cvtColor(img_with_roi, cv2.COLOR_RGB2BGR)
    
    # Draw outer rectangle (white)
    cv2.rectangle(img_bgr, (x_start, y_start), (x_end, y_end), (255, 255, 255), 3)
    
    # Draw inner rectangle (green)
    cv2.rectangle(img_bgr, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # Add text label
    label = f"ROI: {roi_size}x{roi_size}px"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw text background (semi-transparent black)
    text_x = x_start
    text_y = y_start - 10 if y_start > 30 else y_end + 25
    cv2.rectangle(img_bgr, 
                  (text_x, text_y - text_height - 5), 
                  (text_x + text_width + 5, text_y + 5), 
                  (0, 0, 0), -1)
    
    # Draw text (white)
    cv2.putText(img_bgr, label, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb


def create_interactive_roi_selector(image_array, roi_size=100, current_x=0, current_y=0):
    """
    Create an interactive ROI selector using Plotly.
    
    Args:
        image_array: NumPy array of the image (RGB format)
        roi_size: Size of the ROI square
        current_x: Current X offset
        current_y: Current Y offset
    
    Returns:
        Plotly figure object with interactive ROI selection
    """
    height, width = image_array.shape[:2]
    center_y = height // 2
    center_x = width // 2
    
    # Calculate ROI position
    roi_center_x = center_x + current_x
    roi_center_y = center_y + current_y
    half_roi = roi_size // 2
    
    # Calculate ROI boundaries
    y_start = max(0, roi_center_y - half_roi)
    y_end = min(height, roi_center_y + half_roi)
    x_start = max(0, roi_center_x - half_roi)
    x_end = min(width, roi_center_x + half_roi)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add image
    fig.add_trace(go.Image(z=image_array))
    
    # Add ROI rectangle
    fig.add_shape(
        type="rect",
        x0=x_start, y0=y_start,
        x1=x_end, y1=y_end,
        line=dict(color="lime", width=3),
        name="ROI"
    )
    
    # Add center crosshair
    fig.add_shape(
        type="line",
        x0=roi_center_x - 10, y0=roi_center_y,
        x1=roi_center_x + 10, y1=roi_center_y,
        line=dict(color="red", width=2)
    )
    fig.add_shape(
        type="line",
        x0=roi_center_x, y0=roi_center_y - 10,
        x1=roi_center_x, y1=roi_center_y + 10,
        line=dict(color="red", width=2)
    )
    
    # Update layout
    fig.update_layout(
        title=f"คลิกบนภาพเพื่อเลือกตำแหน่ง ROI (ขนาด: {roi_size}x{roi_size}px)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
        width=width if width < 800 else 800,
        height=int((height / width * 800)) if width < 800 else int((height / width * 800)),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='closest'
    )
    
    return fig


def extract_rgb_from_image(image_file, roi_size=100, x_offset=0, y_offset=0):
    """
    Extract average RGB values from a specified region of an image.
    
    This function processes an uploaded image file and calculates the average
    RGB values from a Region of Interest (ROI). The ROI position can be
    customized using x_offset and y_offset parameters.
    
    Process:
        1. Validate the image file format
        2. Read and decode the image using OpenCV
        3. Convert from BGR (OpenCV default) to RGB color space
        4. Calculate ROI position (center + offsets)
        5. Extract ROI from the calculated position
        6. Calculate mean values for each color channel
    
    Args:
        image_file: Uploaded image file object (from Streamlit file_uploader)
                   Must be a file-like object with read() method
        roi_size (int): Size of the square ROI in pixels (default: 100)
                       If image is smaller than roi_size, entire image is used
        x_offset (int): Horizontal offset from center in pixels (default: 0)
                       Negative values move left, positive values move right
        y_offset (int): Vertical offset from center in pixels (default: 0)
                       Negative values move up, positive values move down
    
    Returns:
        tuple: (R, G, B) average values as floats in range [0, 255]
               - R: Red channel average
               - G: Green channel average  
               - B: Blue channel average
        
    Raises:
        ValueError: If image cannot be processed, file is empty, or invalid format
        
    Example:
        >>> with open('sample.jpg', 'rb') as f:
        >>>     # Extract from center
        >>>     r, g, b = extract_rgb_from_image(f, roi_size=100)
        >>>     # Extract from right side (50 pixels right of center)
        >>>     r, g, b = extract_rgb_from_image(f, roi_size=100, x_offset=50)
    """
    try:
        # Validate file format before processing
        if hasattr(image_file, 'name'):
            validate_image_format(image_file.name)
        
        # Convert uploaded file to numpy array for OpenCV processing
        # Read file bytes and convert to uint8 array
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        
        if len(file_bytes) == 0:
            raise ValueError("ไฟล์ภาพว่างเปล่า")
        
        # Decode image from bytes using OpenCV
        # cv2.IMREAD_COLOR loads image in BGR format
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("ไม่สามารถอ่านภาพได้ กรุณาตรวจสอบว่าไฟล์เป็นรูปภาพที่ถูกต้อง")
        
        # Validate image has valid dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("ขนาดภาพไม่ถูกต้อง")
        
        # Convert from BGR (OpenCV default) to RGB color space
        # This is important for correct color representation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions (height, width)
        height, width = image_rgb.shape[:2]
        
        # Calculate center point of the image
        # Using integer division to get pixel coordinates
        center_y = height // 2
        center_x = width // 2
        
        # Apply offsets to center position
        # x_offset: negative = left, positive = right
        # y_offset: negative = up, positive = down
        roi_center_x = center_x + x_offset
        roi_center_y = center_y + y_offset
        
        # Calculate half of ROI size for boundary calculations
        half_roi = roi_size // 2
        
        # Handle case where image is smaller than ROI size
        # In this case, use the entire image for RGB calculation
        if height < roi_size or width < roi_size:
            roi = image_rgb
        else:
            # Extract ROI from specified position
            # Calculate boundaries ensuring they stay within image bounds
            y_start = max(0, roi_center_y - half_roi)
            y_end = min(height, roi_center_y + half_roi)
            x_start = max(0, roi_center_x - half_roi)
            x_end = min(width, roi_center_x + half_roi)
            
            # Slice the image array to extract ROI
            # Format: image[y_start:y_end, x_start:x_end]
            roi = image_rgb[y_start:y_end, x_start:x_end]
        
        # Validate that ROI was successfully extracted
        if roi.size == 0:
            raise ValueError("ไม่สามารถสร้าง ROI จากภาพได้")
        
        # Calculate mean values for each color channel
        # roi[:, :, 0] = Red channel, roi[:, :, 1] = Green, roi[:, :, 2] = Blue
        r_mean = np.mean(roi[:, :, 0])
        g_mean = np.mean(roi[:, :, 1])
        b_mean = np.mean(roi[:, :, 2])
        
        # Validate that calculated values are valid numbers
        if np.isnan(r_mean) or np.isnan(g_mean) or np.isnan(b_mean):
            raise ValueError("ค่า RGB ที่คำนวณได้ไม่ถูกต้อง")
        
        # Return as tuple of floats
        return (float(r_mean), float(g_mean), float(b_mean))
        
    except ValueError as e:
        raise ValueError(str(e))
    except cv2.error as e:
        raise ValueError(f"เกิดข้อผิดพลาดในการประมวลผลภาพด้วย OpenCV: {str(e)}")
    except Exception as e:
        raise ValueError(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")


def save_data_to_csv(r, g, b, concentration, filename='dataset.csv', image_filename=''):
    """
    Save RGB values and concentration to CSV file.
    
    This function saves a single data point (RGB values, concentration, and image filename)
    to a CSV file. If the file doesn't exist, it creates a new file with
    headers. If the file exists, it appends the new data.
    
    CSV Format:
        Image,R,G,B,Concentration
        sample1.jpg,120.5,85.3,45.2,10.5
        sample2.jpg,115.2,80.1,42.8,9.8
        ...
    
    Args:
        r (float): Red channel average value (0-255)
        g (float): Green channel average value (0-255)
        b (float): Blue channel average value (0-255)
        concentration (float): Concentration value in mg/L (>= 0, use 0 for blank)
        filename (str): CSV filename (default: 'dataset.csv')
        image_filename (str): Name of the image file (optional)
    
    Raises:
        ValueError: If input values are invalid (wrong type, out of range)
        IOError: If file operations fail (permission denied, disk full)
        
    Example:
        >>> save_data_to_csv(120.5, 85.3, 45.2, 10.5, 'dataset.csv', 'sample1.jpg')
        >>> # Data saved to dataset.csv with image filename
    """
    try:
        # Validate input values
        if not isinstance(r, (int, float)) or not isinstance(g, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("ค่า RGB ต้องเป็นตัวเลข")
        
        if not isinstance(concentration, (int, float)):
            raise ValueError("ค่าความเข้มข้นต้องเป็นตัวเลข")
        
        if concentration < 0:
            raise ValueError("ค่าความเข้มข้นต้องไม่ติดลบ (ใช้ 0 สำหรับ blank)")
        
        if r < 0 or r > 255 or g < 0 or g > 255 or b < 0 or b > 255:
            raise ValueError("ค่า RGB ต้องอยู่ในช่วง 0-255")
        
        # Check if file exists
        file_exists = os.path.isfile(filename)
        
        if file_exists:
            # Read existing file to check format
            try:
                existing_df = pd.read_csv(filename)
                
                # If old format (no Image column), migrate to new format
                if 'Image' not in existing_df.columns:
                    existing_df.insert(0, 'Image', 'unknown')
                    existing_df.to_csv(filename, index=False)
            except Exception:
                pass  # If can't read, will handle below
        
        # Create DataFrame with new data (including image filename)
        new_data = pd.DataFrame({
            'Image': [image_filename if image_filename else 'unknown'],
            'R': [r],
            'G': [g],
            'B': [b],
            'Concentration': [concentration]
        })
        
        if not file_exists:
            # Create new file with header
            new_data.to_csv(filename, index=False)
        else:
            # Append to existing file
            new_data.to_csv(filename, mode='a', header=False, index=False)
        
        # Verify the data was written successfully
        if not os.path.isfile(filename):
            raise IOError("ไม่สามารถสร้างไฟล์ข้อมูลได้")
            
    except ValueError as e:
        raise ValueError(str(e))
    except IOError as e:
        raise IOError(f"เกิดข้อผิดพลาดในการเขียนไฟล์: {str(e)}")
    except PermissionError:
        raise IOError(f"ไม่มีสิทธิ์เขียนไฟล์ '{filename}' กรุณาตรวจสอบสิทธิ์การเข้าถึง")
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล: {str(e)}")


def main():
    """
    Main function to run the Streamlit application.
    
    This is the entry point of the application. It sets up the page
    configuration, displays the title, and creates a tabbed interface
    for the three main modules:
        1. Data Collection (รวบรวมข้อมูล)
        2. Model Training (เทรนโมเดล)
        3. Prediction (ทำนายผล)
    
    The function is called when the script is run directly.
    """
    st.set_page_config(
        page_title="Colorimetric Analysis Application",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Colorimetric Analysis Application")
    st.write("แอปพลิเคชันวิเคราะห์ค่าความเข้มข้นจากค่าสี RGB")
    
    # Profile selection in sidebar
    st.sidebar.header("⚙️ การตั้งค่า")
    
    # Initialize profile list and units in session state
    if 'profiles' not in st.session_state:
        st.session_state.profiles = ['Phosphate', 'Nitrate', 'Ammonia']
    
    if 'profile_units' not in st.session_state:
        st.session_state.profile_units = {
            'Phosphate': 'mg/L',
            'Nitrate': 'mg/L',
            'Ammonia': 'mg/L'
        }
    
    # Profile selection
    selected_profile = st.sidebar.selectbox(
        "🧪 เลือก Profile สาร",
        st.session_state.profiles,
        help="เลือกชนิดสารที่ต้องการวิเคราะห์ แต่ละ profile จะมี dataset และ model แยกกัน"
    )
    
    # Add new profile
    with st.sidebar.expander("➕ เพิ่ม Profile ใหม่"):
        new_profile = st.text_input("ชื่อ Profile", placeholder="เช่น Iron, Copper", key="new_profile_input")
        new_unit = st.selectbox(
            "หน่วยความเข้มข้น",
            ["mg/L", "ppm", "µg/L", "%", "g/L", "mol/L"],
            key="new_unit_select"
        )
        if st.button("เพิ่ม Profile", key="add_profile_btn"):
            if new_profile and new_profile not in st.session_state.profiles:
                st.session_state.profiles.append(new_profile)
                st.session_state.profile_units[new_profile] = new_unit
                st.success(f"✅ เพิ่ม Profile '{new_profile}' ({new_unit}) สำเร็จ!")
                st.rerun()
            elif new_profile in st.session_state.profiles:
                st.warning("⚠️ Profile นี้มีอยู่แล้ว")
            else:
                st.warning("⚠️ กรุณากรอกชื่อ Profile")
    
    # Delete profile
    with st.sidebar.expander("🗑️ ลบ Profile"):
        if len(st.session_state.profiles) > 1:
            profile_to_delete = st.selectbox(
                "เลือก Profile ที่ต้องการลบ",
                st.session_state.profiles,
                key="delete_profile_select"
            )
            st.warning(f"⚠️ การลบจะไม่ลบไฟล์ dataset และ model")
            if st.button("ลบ Profile", type="secondary", key="delete_profile_btn"):
                if profile_to_delete in st.session_state.profiles:
                    st.session_state.profiles.remove(profile_to_delete)
                    if profile_to_delete in st.session_state.profile_units:
                        del st.session_state.profile_units[profile_to_delete]
                    st.success(f"✅ ลบ Profile '{profile_to_delete}' สำเร็จ!")
                    st.rerun()
        else:
            st.info("ℹ️ ต้องมีอย่างน้อย 1 Profile")
    
    # Edit unit
    with st.sidebar.expander("📏 ปรับหน่วยความเข้มข้น"):
        current_unit = st.session_state.profile_units.get(selected_profile, 'mg/L')
        new_unit = st.selectbox(
            f"หน่วยสำหรับ {selected_profile}",
            ["mg/L", "ppm", "µg/L", "%", "g/L", "mol/L"],
            index=["mg/L", "ppm", "µg/L", "%", "g/L", "mol/L"].index(current_unit) if current_unit in ["mg/L", "ppm", "µg/L", "%", "g/L", "mol/L"] else 0,
            key="edit_unit_select"
        )
        if st.button("บันทึกหน่วย", key="save_unit_btn"):
            st.session_state.profile_units[selected_profile] = new_unit
            st.success(f"✅ เปลี่ยนหน่วยเป็น {new_unit} สำเร็จ!")
            st.rerun()
    
    # Display current profile info
    current_unit = st.session_state.profile_units.get(selected_profile, 'mg/L')
    st.sidebar.info(f"📌 Profile ปัจจุบัน: **{selected_profile}**")
    st.sidebar.write(f"📏 หน่วย: **{current_unit}**")
    st.sidebar.write(f"📁 Dataset: `{selected_profile.lower()}_dataset.csv`")
    st.sidebar.write(f"🤖 Model: `{selected_profile.lower()}_model.joblib`")
    
    # Store selected profile and unit in session state
    st.session_state.current_profile = selected_profile
    st.session_state.current_unit = current_unit
    
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["📊 รวบรวมข้อมูล", "🤖 เทรนโมเดล", "🔮 ทำนายผล"])
    
    with tab1:
        data_collection_module()
    
    with tab2:
        model_training_module()
    
    with tab3:
        prediction_module()


def data_collection_module():
    """
    Data collection module for uploading images and saving RGB data with concentration values.
    
    This module provides a user interface for:
        - Uploading image files (PNG, JPG, JPEG)
        - Displaying the uploaded image
        - Adjusting ROI size manually
        - Automatically calculating RGB values from the image center
        - Inputting the actual concentration value
        - Saving the data point to dataset.csv
    
    The collected data is used later for training the ML model.
    Users should collect at least 10-20 samples for good model accuracy.
    
    UI Components:
        - File uploader for image selection
        - ROI size slider for manual adjustment
        - Image display
        - RGB value metrics (R, G, B)
        - Number input for concentration
        - Save button with validation
    """
    st.header("📊 รวบรวมข้อมูล")
    st.write("อัปโหลดรูปภาพเพื่อคำนวณค่า RGB และบันทึกพร้อมค่าความเข้มข้น")
    
    # ROI configuration
    st.subheader("⚙️ ตั้งค่า ROI (Region of Interest)")
    
    col_roi1, col_roi2 = st.columns(2)
    
    with col_roi1:
        roi_size = st.slider(
            "ขนาด ROI (พิกเซล)",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
            help="ขนาดของพื้นที่สี่เหลี่ยมที่ใช้คำนวณค่า RGB"
        )
    
    with col_roi2:
        roi_position = st.selectbox(
            "ตำแหน่ง ROI",
            ["กลางภาพ (Center)", "กำหนดเอง (Custom)", "คลิกเลือก (Interactive)"],
            help="เลือกตำแหน่งของ ROI บนภาพ"
        )
    
    # Custom position sliders (only show if Custom is selected)
    roi_x_offset = 0
    roi_y_offset = 0
    
    if roi_position == "กำหนดเอง (Custom)":
        st.write("**ปรับตำแหน่ง ROI:**")
        
        # Initialize session state for ROI offsets
        if 'roi_x_offset' not in st.session_state:
            st.session_state.roi_x_offset = 0
        if 'roi_y_offset' not in st.session_state:
            st.session_state.roi_y_offset = 0
        
        # Quick position presets
        st.write("🎯 ตำแหน่งด่วน (คลิกเพื่อเลื่อน ROI):")
        preset_cols = st.columns(5)
        
        with preset_cols[0]:
            if st.button("⬆️ บน", use_container_width=True, help="เลื่อน ROI ไปด้านบน"):
                st.session_state.roi_y_offset = -100
                st.session_state.roi_x_offset = 0
                st.rerun()
        with preset_cols[1]:
            if st.button("⬇️ ล่าง", use_container_width=True, help="เลื่อน ROI ไปด้านล่าง"):
                st.session_state.roi_y_offset = 100
                st.session_state.roi_x_offset = 0
                st.rerun()
        with preset_cols[2]:
            if st.button("⬅️ ซ้าย", use_container_width=True, help="เลื่อน ROI ไปด้านซ้าย"):
                st.session_state.roi_x_offset = -100
                st.session_state.roi_y_offset = 0
                st.rerun()
        with preset_cols[3]:
            if st.button("➡️ ขวา", use_container_width=True, help="เลื่อน ROI ไปด้านขวา"):
                st.session_state.roi_x_offset = 100
                st.session_state.roi_y_offset = 0
                st.rerun()
        with preset_cols[4]:
            if st.button("🎯 กลาง", use_container_width=True, help="รีเซ็ต ROI กลับไปตรงกลาง"):
                st.session_state.roi_x_offset = 0
                st.session_state.roi_y_offset = 0
                st.rerun()
        
        st.write("**ปรับแบบละเอียด (ใช้ slider หรือกดลูกศร):**")
        col_x, col_y = st.columns(2)
        
        with col_x:
            roi_x_offset = st.slider(
                "เลื่อนแนวนอน (X)",
                min_value=-200,
                max_value=200,
                value=st.session_state.roi_x_offset,
                step=5,
                help="เลื่อน ROI ไปทางซ้าย (-) หรือขวา (+) จากจุดกลาง",
                key="slider_x"
            )
            st.session_state.roi_x_offset = roi_x_offset
        
        with col_y:
            roi_y_offset = st.slider(
                "เลื่อนแนวตั้ง (Y)",
                min_value=-200,
                max_value=200,
                value=st.session_state.roi_y_offset,
                step=5,
                help="เลื่อน ROI ไปทางบน (-) หรือล่าง (+) จากจุดกลาง",
                key="slider_y"
            )
            st.session_state.roi_y_offset = roi_y_offset
    
    st.info(f"📐 ขนาด ROI: {roi_size}x{roi_size} พิกเซล | ตำแหน่ง: {roi_position}")
    
    st.divider()
    
    # File uploader for image
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปภาพ", 
        type=['png', 'jpg', 'jpeg'],
        help="รองรับไฟล์ PNG, JPG, และ JPEG",
        key="data_collection_uploader"
    )
    
    # Store uploaded file in session state to persist across reruns
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
    
    # Use the stored file if available
    if 'uploaded_image' in st.session_state and st.session_state.uploaded_image is not None:
        try:
            # Create two-column layout for image and analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("รูปภาพที่อัปโหลด")
                # Display image using PIL (Pillow) for better Streamlit integration
                image = Image.open(st.session_state.uploaded_image)
                
                # Convert PIL image to numpy array for ROI drawing
                image_array = np.array(image)
                
                # Interactive ROI selection mode
                if roi_position == "คลิกเลือก (Interactive)":
                    # Initialize session state for interactive selection
                    if 'interactive_x' not in st.session_state:
                        st.session_state.interactive_x = 0
                    if 'interactive_y' not in st.session_state:
                        st.session_state.interactive_y = 0
                    
                    # Create interactive Plotly figure
                    fig = create_interactive_roi_selector(
                        image_array,
                        roi_size=roi_size,
                        current_x=st.session_state.interactive_x,
                        current_y=st.session_state.interactive_y
                    )
                    
                    # Display interactive figure
                    selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="roi_selector")
                    
                    # Handle click events
                    if selected_points and 'selection' in selected_points:
                        if 'points' in selected_points['selection'] and len(selected_points['selection']['points']) > 0:
                            point = selected_points['selection']['points'][0]
                            if 'x' in point and 'y' in point:
                                # Calculate offset from center
                                height, width = image_array.shape[:2]
                                center_x = width // 2
                                center_y = height // 2
                                st.session_state.interactive_x = int(point['x'] - center_x)
                                st.session_state.interactive_y = int(point['y'] - center_y)
                                st.rerun()
                    
                    # Use interactive offsets
                    roi_x_offset = st.session_state.interactive_x
                    roi_y_offset = st.session_state.interactive_y
                    
                    st.info(f"📍 ตำแหน่ง: X={roi_x_offset:+d}, Y={roi_y_offset:+d} จากจุดกลาง")
                else:
                    # Draw ROI rectangle on image (non-interactive mode)
                    image_with_roi = draw_roi_on_image(
                        image_array, 
                        roi_size=roi_size,
                        x_offset=roi_x_offset,
                        y_offset=roi_y_offset
                    )
                    
                    # Display image with ROI
                    st.image(image_with_roi, use_container_width=True, caption="กรอบสีเขียว = พื้นที่ ROI ที่ใช้วิเคราะห์")
            
            with col2:
                st.subheader("ผลการวิเคราะห์")
                
                # Reset file pointer to beginning for OpenCV processing
                # PIL already read the file, so we need to reset for OpenCV
                st.session_state.uploaded_image.seek(0)
                
                # Extract RGB values with custom ROI size and position
                try:
                    r, g, b = extract_rgb_from_image(
                        st.session_state.uploaded_image, 
                        roi_size=roi_size,
                        x_offset=roi_x_offset,
                        y_offset=roi_y_offset
                    )
                    
                    # Display RGB values
                    st.success("คำนวณค่า RGB สำเร็จ!")
                    
                    # Create metrics display
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Red (R)", f"{r:.2f}")
                    with metric_col2:
                        st.metric("Green (G)", f"{g:.2f}")
                    with metric_col3:
                        st.metric("Blue (B)", f"{b:.2f}")
                    
                    st.divider()
                    
                    # Input for concentration value
                    st.subheader("บันทึกข้อมูล")
                    unit = st.session_state.get('current_unit', 'mg/L')
                    concentration = st.number_input(
                        f"ค่าความเข้มข้น ({unit})",
                        min_value=0.0,
                        step=0.1,
                        format="%.2f",
                        help=f"กรอกค่าความเข้มข้นที่วัดได้จริงในหน่วย {unit} (ใช้ 0 สำหรับตัวอย่าง blank)"
                    )
                    
                    # Save button
                    if st.button("💾 บันทึกข้อมูล", type="primary"):
                        if concentration >= 0:
                            try:
                                dataset_file, _ = get_profile_files()
                                # Get image filename
                                img_name = st.session_state.uploaded_image.name if hasattr(st.session_state.uploaded_image, 'name') else 'unknown'
                                save_data_to_csv(r, g, b, concentration, dataset_file, img_name)
                                st.success("✅ บันทึกข้อมูลสำเร็จ!")
                                st.balloons()
                            except ValueError as e:
                                st.error(f"❌ ข้อมูลไม่ถูกต้อง: {str(e)}")
                            except IOError as e:
                                st.error(f"❌ ไม่สามารถบันทึกไฟล์ได้: {str(e)}")
                            except Exception as e:
                                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
                        else:
                            st.warning("⚠️ กรุณากรอกค่าความเข้มข้น (ใช้ 0 สำหรับ blank)")
                    
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลภาพ: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ ไม่สามารถแสดงภาพได้: {str(e)}")
    else:
        st.info("👆 กรุณาอัปโหลดรูปภาพเพื่อเริ่มต้น")


def train_model(dataset_file='dataset.csv', model_file='model.joblib'):
    """
    Train a RandomForestRegressor model and save it.
    
    This function loads data from a CSV file, trains a Random Forest
    Regressor model to predict concentration from RGB values, and saves
    the trained model to a file.
    
    Process:
        1. Load and validate dataset from CSV
        2. Separate features (R, G, B) and target (Concentration)
        3. Train RandomForestRegressor with 100 estimators
        4. Calculate R² score to measure model performance
        5. Save trained model using joblib
    
    Model Configuration:
        - Algorithm: RandomForestRegressor
        - n_estimators: 100 (number of decision trees)
        - random_state: 42 (for reproducibility)
    
    Args:
        dataset_file (str): Path to the CSV dataset file (default: 'dataset.csv')
                           Must contain columns: R, G, B, Concentration
        model_file (str): Path to save the trained model (default: 'model.joblib')
    
    Returns:
        tuple: (model, r2_score)
            - model: Trained RandomForestRegressor object
            - r2_score (float): R² score (0-1), higher is better
                              0.9-1.0: Excellent
                              0.7-0.9: Good
                              0.5-0.7: Moderate
                              <0.5: Poor (need more data)
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset has insufficient data (<5 samples) or invalid format
        IOError: If model file cannot be saved (permission denied)
        Exception: If training fails for other reasons
        
    Example:
        >>> model, r2 = train_model('dataset.csv', 'model.joblib')
        >>> print(f"Model trained with R² score: {r2:.4f}")
    """
    try:
        # Check if dataset file exists
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล '{dataset_file}' กรุณารวบรวมข้อมูลก่อน")
        
        # Check file size
        if os.path.getsize(dataset_file) == 0:
            raise ValueError("ไฟล์ข้อมูลว่างเปล่า กรุณารวบรวมข้อมูลก่อน")
        
        # Load data from CSV with format detection
        try:
            df = pd.read_csv(dataset_file)
            
            # Check if Image column exists, if not add it
            if 'Image' not in df.columns:
                df.insert(0, 'Image', 'unknown')
                # Save migrated format
                df.to_csv(dataset_file, index=False)
                
        except pd.errors.EmptyDataError:
            raise ValueError("ไฟล์ข้อมูลว่างเปล่า กรุณารวบรวมข้อมูลก่อน")
        except pd.errors.ParserError as e:
            raise ValueError(f"ไฟล์ข้อมูลมีรูปแบบไม่ถูกต้อง: {str(e)}")
        
        # Validate that CSV has all required columns
        # Dataset must have R, G, B (features) and Concentration (target)
        required_columns = ['R', 'G', 'B', 'Concentration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"ไฟล์ข้อมูลขาดคอลัมน์: {', '.join(missing_columns)}")
        
        # Check if dataset has sufficient data for training
        # Minimum 5 samples required, but 10-20+ recommended for good accuracy
        if len(df) < 5:
            raise ValueError(f"ข้อมูลไม่เพียงพอ (มี {len(df)} แถว) กรุณารวบรวมข้อมูลอย่างน้อย 5 ตัวอย่าง")
        
        # Check for missing (NaN) values in any required column
        # Missing values would cause training to fail
        if df[required_columns].isnull().any().any():
            raise ValueError("พบข้อมูลที่หายไปในไฟล์ กรุณาตรวจสอบข้อมูล")
        
        # Validate data types and value ranges for each column
        # RGB values must be numeric and in range [0, 255]
        for col in ['R', 'G', 'B']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"คอลัมน์ {col} ต้องเป็นตัวเลข")
            # Check that all RGB values are within valid range
            if (df[col] < 0).any() or (df[col] > 255).any():
                raise ValueError(f"คอลัมน์ {col} มีค่าที่ไม่อยู่ในช่วง 0-255")
        
        # Validate concentration column
        if not pd.api.types.is_numeric_dtype(df['Concentration']):
            raise ValueError("คอลัมน์ Concentration ต้องเป็นตัวเลข")
        
        # Concentration must be non-negative (>= 0, where 0 = blank)
        if (df['Concentration'] < 0).any():
            raise ValueError("คอลัมน์ Concentration มีค่าติดลบ (ใช้ 0 สำหรับ blank)")
        
        # Separate features (R, G, B) and target (Concentration)
        X = df[['R', 'G', 'B']]
        y = df['Concentration']
        
        # Create and train RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate R² score
        r2_score = model.score(X, y)
        
        # Validate R² score
        if np.isnan(r2_score) or np.isinf(r2_score):
            raise ValueError("ไม่สามารถคำนวณ R² score ได้ กรุณาตรวจสอบข้อมูล")
        
        # Save model to file
        try:
            joblib.dump(model, model_file)
        except Exception as e:
            raise IOError(f"ไม่สามารถบันทึกโมเดลได้: {str(e)}")
        
        # Verify model file was created
        if not os.path.isfile(model_file):
            raise IOError("ไม่สามารถสร้างไฟล์โมเดลได้")
        
        return model, r2_score
        
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except ValueError as e:
        raise ValueError(str(e))
    except IOError as e:
        raise IOError(str(e))
    except PermissionError:
        raise IOError(f"ไม่มีสิทธิ์เขียนไฟล์ '{model_file}' กรุณาตรวจสอบสิทธิ์การเข้าถึง")
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการเทรนโมเดล: {str(e)}")


def calculate_lod(df, model):
    """
    Calculate Limit of Detection (LOD) from blank samples.
    
    LOD is calculated as: LOD = mean(blank) + 3 × SD(blank predictions)
    
    Args:
        df: DataFrame with columns R, G, B, Concentration
        model: Trained model
    
    Returns:
        float: LOD value, or None if no blank samples
    """
    try:
        # Get blank samples (concentration = 0)
        blank_samples = df[df['Concentration'] == 0.0]
        
        if len(blank_samples) < 3:
            return None  # Need at least 3 blanks for reliable LOD
        
        # Predict concentrations for blank samples
        X_blank = blank_samples[['R', 'G', 'B']]
        blank_predictions = model.predict(X_blank)
        
        # Calculate LOD = mean + 3*SD
        mean_blank = np.mean(blank_predictions)
        sd_blank = np.std(blank_predictions)
        lod = mean_blank + (3 * sd_blank)
        
        return max(0, lod)  # LOD cannot be negative
        
    except Exception:
        return None


def plot_calibration_curve(df, model, unit='mg/L'):
    """
    Create calibration curve plot showing actual vs predicted concentrations.
    
    Args:
        df: DataFrame with columns R, G, B, Concentration
        model: Trained model
        unit: Unit of concentration (default: 'mg/L')
    
    Returns:
        matplotlib figure object
    """
    try:
        # Prepare data
        X = df[['R', 'G', 'B']]
        y_actual = df['Concentration']
        y_predicted = model.predict(X)
        
        # Calculate R² score
        r2 = r2_score(y_actual, y_predicted)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual data points
        ax.scatter(y_actual, y_predicted, alpha=0.6, s=100, edgecolors='black', linewidth=1.5)
        
        # Plot ideal line (y=x)
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
        
        # Labels and title
        ax.set_xlabel(f'Actual Concentration ({unit})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted Concentration ({unit})', fontsize=12, fontweight='bold')
        ax.set_title(f'Calibration Curve\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(fontsize=10)
        
        # Add R² text box
        textstr = f'R² = {r2:.4f}\nn = {len(df)} samples'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"ไม่สามารถสร้างกราฟได้: {str(e)}")
        return None


def model_training_module():
    """
    Model training module for training ML model from collected data.
    
    This module provides a user interface for:
        - Viewing dataset information (number of samples)
        - Previewing the collected data
        - Training a Random Forest Regressor model
        - Displaying model performance (R² score)
        - Displaying calibration curve
        - Calculating and displaying LOD (Limit of Detection)
        - Saving the trained model to model.joblib
    
    The module validates that sufficient data exists (minimum 5 samples)
    before training. It provides feedback on model accuracy and suggests
    collecting more data if accuracy is low.
    
    UI Components:
        - Dataset information display
        - Data preview (expandable)
        - Train button
        - R² score metrics
        - Accuracy interpretation messages
    """
    st.header("🤖 เทรนโมเดล")
    
    # Get profile-specific files
    dataset_file, model_file = get_profile_files()
    profile_name = st.session_state.get('current_profile', 'Default')
    
    st.write(f"เทรนโมเดล Machine Learning สำหรับ **{profile_name}** จากข้อมูลที่รวบรวมไว้")
    
    # Check if dataset exists
    if os.path.isfile(dataset_file):
        # Show dataset info
        try:
            # Try to read CSV - if it fails, it might be old format
            df = pd.read_csv(dataset_file)
            
            # Check if Image column exists, if not add it
            if 'Image' not in df.columns:
                df.insert(0, 'Image', 'unknown')
                # Save the migrated format back to file
                df.to_csv(dataset_file, index=False)
            st.info(f"📊 พบข้อมูล {len(df)} ตัวอย่างในไฟล์ {dataset_file}")
            
            # Data management section
            with st.expander("📋 จัดการข้อมูล", expanded=False):
                st.write("**ดูและแก้ไขข้อมูล**")
                
                # Add index column for selection
                df_display = df.copy()
                df_display.insert(0, 'ลำดับ', range(1, len(df) + 1))
                
                # Reorder columns to show Image first
                cols = ['ลำดับ', 'Image', 'R', 'G', 'B', 'Concentration']
                df_display = df_display[[c for c in cols if c in df_display.columns]]
                
                # Display dataframe with selection (hide index)
                st.dataframe(df_display, use_container_width=True, height=300, hide_index=True)
                
                st.divider()
                
                # Delete data section
                st.write("**ลบข้อมูล**")
                col_del1, col_del2 = st.columns([3, 1])
                
                with col_del1:
                    # Multi-select for rows to delete
                    rows_to_delete = st.multiselect(
                        "เลือกลำดับที่ต้องการลบ",
                        options=list(range(1, len(df) + 1)),
                        help="เลือกได้หลายแถว กด Ctrl/Cmd + Click"
                    )
                
                with col_del2:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    if st.button("🗑️ ลบข้อมูลที่เลือก", type="secondary"):
                        if rows_to_delete:
                            # Convert to 0-based index
                            indices_to_delete = [i - 1 for i in rows_to_delete]
                            
                            # Check if deleting would leave too few samples
                            remaining = len(df) - len(indices_to_delete)
                            if remaining < 5:
                                st.error(f"❌ ไม่สามารถลบได้ เหลือข้อมูลเพียง {remaining} ตัวอย่าง (ต้องมีอย่างน้อย 5)")
                            else:
                                # Delete rows
                                df_new = df.drop(indices_to_delete).reset_index(drop=True)
                                
                                # Save back to CSV
                                df_new.to_csv(dataset_file, index=False)
                                
                                st.success(f"✅ ลบข้อมูล {len(indices_to_delete)} แถวสำเร็จ! เหลือ {len(df_new)} ตัวอย่าง")
                                st.rerun()
                        else:
                            st.warning("⚠️ กรุณาเลือกข้อมูลที่ต้องการลบ")
                
                st.divider()
                
                # Clear all data
                st.write("**ล้างข้อมูลทั้งหมด**")
                st.warning("⚠️ การล้างข้อมูลจะลบไฟล์ทั้งหมด ไม่สามารถกู้คืนได้!")
                
                col_clear1, col_clear2, col_clear3 = st.columns([2, 1, 1])
                with col_clear2:
                    confirm_clear = st.checkbox("ยืนยันการล้างข้อมูล")
                with col_clear3:
                    if st.button("🗑️ ล้างทั้งหมด", type="secondary", disabled=not confirm_clear):
                        try:
                            os.remove(dataset_file)
                            st.success("✅ ล้างข้อมูลทั้งหมดสำเร็จ!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ ไม่สามารถล้างข้อมูลได้: {str(e)}")
            
            # Show preview of dataset
            with st.expander("📊 สถิติข้อมูล"):
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("จำนวนตัวอย่าง", len(df))
                with col_stat2:
                    st.metric("ความเข้มข้นต่ำสุด", f"{df['Concentration'].min():.2f}")
                with col_stat3:
                    st.metric("ความเข้มข้นสูงสุด", f"{df['Concentration'].max():.2f}")
                with col_stat4:
                    st.metric("ความเข้มข้นเฉลี่ย", f"{df['Concentration'].mean():.2f}")
                
                st.write("**สถิติค่า RGB และความเข้มข้น:**")
                numeric_cols = ['R', 'G', 'B', 'Concentration']
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถอ่านข้อมูลได้: {str(e)}")
    else:
        st.warning("⚠️ ยังไม่พบไฟล์ข้อมูล กรุณารวบรวมข้อมูลก่อน")
    
    st.divider()
    
    # Train button
    if st.button("🚀 เริ่มเทรนโมเดล", type="primary"):
        try:
            with st.spinner("กำลังเทรนโมเดล..."):
                # Train model with profile-specific files
                model, r2_score = train_model(dataset_file, model_file)
                
                # Display results
                st.success("✅ เทรนโมเดลสำเร็จ!")
                st.balloons()
                
                # Show R² score
                st.subheader("ผลการเทรน")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "R² Score", 
                        f"{r2_score:.4f}",
                        help="ค่า R² ยิ่งใกล้ 1 แสดงว่าโมเดลทำนายได้แม่นยำ"
                    )
                
                with col2:
                    accuracy_percent = r2_score * 100
                    st.metric(
                        "ความแม่นยำ",
                        f"{accuracy_percent:.2f}%"
                    )
                
                # Show interpretation
                if r2_score >= 0.9:
                    st.success("🎯 โมเดลมีความแม่นยำสูงมาก!")
                elif r2_score >= 0.7:
                    st.info("👍 โมเดลมีความแม่นยำดี")
                elif r2_score >= 0.5:
                    st.warning("⚠️ โมเดลมีความแม่นยำปานกลาง ควรรวบรวมข้อมูลเพิ่มเติม")
                else:
                    st.error("❌ โมเดลมีความแม่นยำต่ำ กรุณารวบรวมข้อมูลเพิ่มเติม")
                
                st.info(f"💾 โมเดลถูกบันทึกเป็นไฟล์ '{model_file}' แล้ว")
                
                st.divider()
                
                # Load dataset for plotting
                df_plot = pd.read_csv(dataset_file)
                if 'Image' not in df_plot.columns:
                    df_plot.insert(0, 'Image', 'unknown')
                
                # Calculate LOD
                lod_value = calculate_lod(df_plot, model)
                
                # Display LOD
                st.subheader("📊 Limit of Detection (LOD)")
                if lod_value is not None:
                    unit = st.session_state.get('current_unit', 'mg/L')
                    col_lod1, col_lod2 = st.columns(2)
                    
                    with col_lod1:
                        st.metric(
                            "LOD", 
                            f"{lod_value:.4f} {unit}",
                            help="ค่าความเข้มข้นต่ำสุดที่สามารถตรวจวัดได้อย่างน่าเชื่อถือ"
                        )
                    
                    with col_lod2:
                        blank_count = len(df_plot[df_plot['Concentration'] == 0.0])
                        st.metric(
                            "จำนวน Blank", 
                            f"{blank_count} ตัวอย่าง",
                            help="จำนวนตัวอย่าง blank ที่ใช้คำนวณ LOD"
                        )
                    
                    st.info(f"💡 LOD คำนวณจาก: LOD = mean(blank) + 3×SD(blank predictions)")
                else:
                    st.warning("⚠️ ไม่สามารถคำนวณ LOD ได้ (ต้องมีตัวอย่าง blank อย่างน้อย 3 ตัวอย่าง)")
                
                st.divider()
                
                # Plot calibration curve
                st.subheader("📈 Calibration Curve")
                unit = st.session_state.get('current_unit', 'mg/L')
                fig = plot_calibration_curve(df_plot, model, unit)
                
                if fig is not None:
                    st.pyplot(fig)
                    st.success("✅ กราฟ Calibration Curve แสดงความสัมพันธ์ระหว่างค่าจริงและค่าทำนาย")
                    st.info("💡 จุดที่อยู่ใกล้เส้นแดง (y=x) แสดงว่าโมเดลทำนายได้แม่นยำ")
                
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.info("💡 กรุณาไปที่แท็บ 'รวบรวมข้อมูล' เพื่อเริ่มรวบรวมข้อมูล")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            if "ไม่เพียงพอ" in str(e):
                st.info("💡 แนะนำให้รวบรวมข้อมูลอย่างน้อย 10-20 ตัวอย่างเพื่อความแม่นยำที่ดี")
        except IOError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")


def load_model(model_file='model.joblib'):
    """
    Load a trained model from file.
    
    This function loads a previously trained and saved machine learning
    model from a joblib file. The model must have been saved using the
    train_model() function or joblib.dump().
    
    Args:
        model_file (str): Path to the model file (default: 'model.joblib')
    
    Returns:
        model: Trained model object (typically RandomForestRegressor)
               Ready to use for predictions with predict() method
        
    Raises:
        FileNotFoundError: If model file doesn't exist (need to train first)
        ValueError: If model file is corrupted or invalid format
        IOError: If file cannot be read (permission denied)
        Exception: If model loading fails for other reasons
        
    Example:
        >>> model = load_model('model.joblib')
        >>> prediction = model.predict([[120, 85, 45]])
        >>> print(f"Predicted concentration: {prediction[0]:.2f} mg/L")
    """
    try:
        # Check if model file exists
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"ไม่พบโมเดล กรุณาเทรนโมเดลก่อน")
        
        # Check file size
        if os.path.getsize(model_file) == 0:
            raise ValueError("ไฟล์โมเดลว่างเปล่า กรุณาเทรนโมเดลใหม่")
        
        # Load model from file
        try:
            model = joblib.load(model_file)
        except Exception as e:
            raise ValueError(f"ไฟล์โมเดลเสียหาย กรุณาเทรนโมเดลใหม่: {str(e)}")
        
        # Validate model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError("โมเดลไม่ถูกต้อง ไม่มีฟังก์ชัน predict")
        
        return model
        
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))
    except ValueError as e:
        raise ValueError(str(e))
    except PermissionError:
        raise IOError(f"ไม่มีสิทธิ์อ่านไฟล์ '{model_file}' กรุณาตรวจสอบสิทธิ์การเข้าถึง")
    except Exception as e:
        raise Exception(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")


def prediction_module():
    """
    Prediction module for predicting concentration from new images.
    
    This module provides a user interface for:
        - Uploading new images for prediction
        - Displaying the uploaded image
        - Adjusting ROI size manually
        - Loading the trained model
        - Extracting RGB values from the image
        - Predicting concentration using the model
        - Displaying the prediction result
    
    The module requires a trained model (model.joblib) to exist.
    If no model is found, it instructs the user to train one first.
    
    UI Components:
        - ROI size slider for manual adjustment
        - File uploader for new images (with unique key 'predict')
        - Image display
        - RGB value metrics
        - Predicted concentration display (large format)
        - Model status indicator
        - Error handling and user guidance
    """
    st.header("🔮 ทำนายผล")
    
    # Get profile-specific files
    dataset_file, model_file = get_profile_files()
    profile_name = st.session_state.get('current_profile', 'Default')
    
    st.write(f"อัปโหลดรูปภาพใหม่เพื่อทำนายค่าความเข้มข้นของ **{profile_name}**")
    
    # ROI configuration
    st.subheader("⚙️ ตั้งค่า ROI (Region of Interest)")
    
    col_roi1, col_roi2 = st.columns(2)
    
    with col_roi1:
        roi_size = st.slider(
            "ขนาด ROI (พิกเซล)",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
            help="ขนาดของพื้นที่สี่เหลี่ยมที่ใช้คำนวณค่า RGB",
            key="predict_roi_size"
        )
    
    with col_roi2:
        roi_position = st.selectbox(
            "ตำแหน่ง ROI",
            ["กลางภาพ (Center)", "กำหนดเอง (Custom)", "คลิกเลือก (Interactive)"],
            help="เลือกตำแหน่งของ ROI บนภาพ",
            key="predict_roi_position"
        )
    
    # Initialize ROI offsets
    roi_x_offset = 0
    roi_y_offset = 0
    
    # Custom position sliders (only show if Custom is selected)
    if roi_position == "กำหนดเอง (Custom)":
        st.write("**ปรับตำแหน่ง ROI:**")
        
        # Initialize session state for prediction ROI offsets
        if 'predict_roi_x_offset' not in st.session_state:
            st.session_state.predict_roi_x_offset = 0
        if 'predict_roi_y_offset' not in st.session_state:
            st.session_state.predict_roi_y_offset = 0
        
        # Quick position presets
        st.write("🎯 ตำแหน่งด่วน (คลิกเพื่อเลื่อน ROI):")
        preset_cols = st.columns(5)
        
        with preset_cols[0]:
            if st.button("⬆️ บน", use_container_width=True, help="เลื่อน ROI ไปด้านบน", key="pred_btn_top"):
                st.session_state.predict_roi_y_offset = -100
                st.session_state.predict_roi_x_offset = 0
                st.rerun()
        with preset_cols[1]:
            if st.button("⬇️ ล่าง", use_container_width=True, help="เลื่อน ROI ไปด้านล่าง", key="pred_btn_bottom"):
                st.session_state.predict_roi_y_offset = 100
                st.session_state.predict_roi_x_offset = 0
                st.rerun()
        with preset_cols[2]:
            if st.button("⬅️ ซ้าย", use_container_width=True, help="เลื่อน ROI ไปด้านซ้าย", key="pred_btn_left"):
                st.session_state.predict_roi_x_offset = -100
                st.session_state.predict_roi_y_offset = 0
                st.rerun()
        with preset_cols[3]:
            if st.button("➡️ ขวา", use_container_width=True, help="เลื่อน ROI ไปด้านขวา", key="pred_btn_right"):
                st.session_state.predict_roi_x_offset = 100
                st.session_state.predict_roi_y_offset = 0
                st.rerun()
        with preset_cols[4]:
            if st.button("🎯 กลาง", use_container_width=True, help="รีเซ็ต ROI กลับไปตรงกลาง", key="pred_btn_center"):
                st.session_state.predict_roi_x_offset = 0
                st.session_state.predict_roi_y_offset = 0
                st.rerun()
        
        st.write("**ปรับแบบละเอียด (ใช้ slider หรือกดลูกศร):**")
        col_x, col_y = st.columns(2)
        
        with col_x:
            roi_x_offset = st.slider(
                "เลื่อนแนวนอน (X)",
                min_value=-200,
                max_value=200,
                value=st.session_state.predict_roi_x_offset,
                step=5,
                help="เลื่อน ROI ไปทางซ้าย (-) หรือขวา (+) จากจุดกลาง",
                key="predict_slider_x"
            )
            st.session_state.predict_roi_x_offset = roi_x_offset
        
        with col_y:
            roi_y_offset = st.slider(
                "เลื่อนแนวตั้ง (Y)",
                min_value=-200,
                max_value=200,
                value=st.session_state.predict_roi_y_offset,
                step=5,
                help="เลื่อน ROI ไปทางบน (-) หรือล่าง (+) จากจุดกลาง",
                key="predict_slider_y"
            )
            st.session_state.predict_roi_y_offset = roi_y_offset
    
    st.info(f"📐 ขนาด ROI: {roi_size}x{roi_size} พิกเซล | ตำแหน่ง: {roi_position}")
    st.warning("⚠️ แนะนำให้ใช้การตั้งค่า ROI เดียวกับที่ใช้ตอนรวบรวมข้อมูลเพื่อความแม่นยำ")
    
    st.divider()
    
    # File uploader for new image (with unique key)
    uploaded_file = st.file_uploader(
        "อัปโหลดรูปภาพใหม่", 
        type=['png', 'jpg', 'jpeg'],
        key='predict_uploader',
        help="รองรับไฟล์ PNG, JPG, และ JPEG"
    )
    
    # Store uploaded file in session state to persist across reruns
    if uploaded_file is not None:
        st.session_state.predict_uploaded_image = uploaded_file
    
    # Use the stored file if available
    if 'predict_uploaded_image' in st.session_state and st.session_state.predict_uploaded_image is not None:
        try:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("รูปภาพที่อัปโหลด")
                # Display image using PIL
                image = Image.open(st.session_state.predict_uploaded_image)
                
                # Convert PIL image to numpy array for ROI drawing
                image_array = np.array(image)
                
                # Interactive ROI selection mode
                if roi_position == "คลิกเลือก (Interactive)":
                    # Initialize session state for interactive selection
                    if 'predict_interactive_x' not in st.session_state:
                        st.session_state.predict_interactive_x = 0
                    if 'predict_interactive_y' not in st.session_state:
                        st.session_state.predict_interactive_y = 0
                    
                    # Create interactive Plotly figure
                    fig = create_interactive_roi_selector(
                        image_array,
                        roi_size=roi_size,
                        current_x=st.session_state.predict_interactive_x,
                        current_y=st.session_state.predict_interactive_y
                    )
                    
                    # Display interactive figure
                    selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="predict_roi_selector")
                    
                    # Handle click events
                    if selected_points and 'selection' in selected_points:
                        if 'points' in selected_points['selection'] and len(selected_points['selection']['points']) > 0:
                            point = selected_points['selection']['points'][0]
                            if 'x' in point and 'y' in point:
                                # Calculate offset from center
                                height, width = image_array.shape[:2]
                                center_x = width // 2
                                center_y = height // 2
                                st.session_state.predict_interactive_x = int(point['x'] - center_x)
                                st.session_state.predict_interactive_y = int(point['y'] - center_y)
                                st.rerun()
                    
                    # Use interactive offsets
                    roi_x_offset = st.session_state.predict_interactive_x
                    roi_y_offset = st.session_state.predict_interactive_y
                    
                    st.info(f"📍 ตำแหน่ง: X={roi_x_offset:+d}, Y={roi_y_offset:+d} จากจุดกลาง")
                else:
                    # Draw ROI rectangle on image (non-interactive mode)
                    image_with_roi = draw_roi_on_image(
                        image_array, 
                        roi_size=roi_size,
                        x_offset=roi_x_offset,
                        y_offset=roi_y_offset
                    )
                    
                    # Display image with ROI
                    st.image(image_with_roi, use_container_width=True, caption="กรอบสีเขียว = พื้นที่ ROI ที่ใช้วิเคราะห์")
            
            with col2:
                st.subheader("ผลการทำนาย")
                
                try:
                    # Load model for current profile
                    model = load_model(model_file)
                    
                    # Reset file pointer for OpenCV processing
                    st.session_state.predict_uploaded_image.seek(0)
                    
                    # Extract RGB values from image with custom ROI size and position
                    r, g, b = extract_rgb_from_image(
                        st.session_state.predict_uploaded_image, 
                        roi_size=roi_size,
                        x_offset=roi_x_offset,
                        y_offset=roi_y_offset
                    )
                    
                    # Display RGB values
                    st.info("ค่า RGB ที่วิเคราะห์ได้:")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Red (R)", f"{r:.2f}")
                    with metric_col2:
                        st.metric("Green (G)", f"{g:.2f}")
                    with metric_col3:
                        st.metric("Blue (B)", f"{b:.2f}")
                    
                    st.divider()
                    
                    # Prepare features for prediction
                    # Model expects 2D array with shape (n_samples, n_features)
                    # We have 1 sample with 3 features (R, G, B)
                    features = np.array([[r, g, b]])
                    
                    # Predict concentration using the trained model
                    # predict() returns array, we take first element [0]
                    predicted_concentration = model.predict(features)[0]
                    
                    # Display prediction result
                    st.success("✅ ทำนายค่าความเข้มข้นสำเร็จ!")
                    
                    # Show predicted concentration with large display
                    unit = st.session_state.get('current_unit', 'mg/L')
                    st.markdown("### ผลการทำนาย")
                    st.markdown(f"## 🎯 {predicted_concentration:.2f} {unit}")
                    
                    # Additional info
                    st.info(f"💡 ค่าที่แสดงคือค่าความเข้มข้นในหน่วย {unit} ที่ทำนายโดยโมเดล Machine Learning")
                    
                except FileNotFoundError as e:
                    st.error(f"❌ {str(e)}")
                    st.info("👉 กรุณาไปที่แท็บ 'เทรนโมเดล' เพื่อเทรนโมเดลก่อน")
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                    if "โมเดล" in str(e):
                        st.info("👉 กรุณาไปที่แท็บ 'เทรนโมเดล' เพื่อเทรนโมเดลใหม่")
                except IOError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ ไม่สามารถแสดงภาพได้: {str(e)}")
    else:
        st.info("👆 กรุณาอัปโหลดรูปภาพเพื่อเริ่มทำนาย")
        
        # Show model status
        if os.path.isfile(model_file):
            st.success(f"✅ พบโมเดลที่เทรนแล้ว ({model_file}) พร้อมใช้งาน")
        else:
            st.warning(f"⚠️ ยังไม่พบโมเดลสำหรับ {profile_name} กรุณาเทรนโมเดลก่อนใช้งาน")


if __name__ == "__main__":
    main()
