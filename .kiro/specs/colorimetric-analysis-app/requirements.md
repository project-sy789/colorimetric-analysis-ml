# Requirements Document

## Introduction

แอปพลิเคชันนี้เป็นเว็บแอปพลิเคชันที่พัฒนาด้วย Python และ Streamlit สำหรับการวิเคราะห์เชิงสี (Colorimetric Analysis) เพื่อเทรนโมเดล Machine Learning ในการทำนายค่าความเข้มข้นของสาร (เช่น ฟอสเฟต) จากค่าสี RGB ที่วิเคราะห์จากรูปภาพ แอปพลิเคชันช่วยให้ผู้ใช้สามารถรวบรวมข้อมูล เทรนโมเดล และทำนายค่าความเข้มข้นจากภาพใหม่ได้อย่างง่ายดาย

## Glossary

- **Application**: แอปพลิเคชัน Streamlit สำหรับการวิเคราะห์เชิงสี
- **User**: ผู้ใช้งานแอปพลิเคชัน
- **RGB Values**: ค่าสีแดง (R), เขียว (G), น้ำเงิน (B) ที่คำนวณจากภาพ
- **ROI (Region of Interest)**: พื้นที่สนใจในภาพที่ใช้คำนวณค่าสี
- **Dataset File**: ไฟล์ dataset.csv ที่เก็บข้อมูล RGB และค่าความเข้มข้น
- **Model File**: ไฟล์ model.joblib ที่เก็บโมเดล Machine Learning ที่เทรนแล้ว
- **Concentration Value**: ค่าความเข้มข้นของสาร (เช่น mg/L)
- **ML Model**: โมเดล Machine Learning (RandomForestRegressor หรือ LinearRegression)

## Requirements

### Requirement 1: การจัดการส่วนหลักของแอปพลิเคชัน

**User Story:** ในฐานะผู้ใช้ ฉันต้องการเข้าถึงส่วนต่างๆ ของแอปพลิเคชันได้อย่างชัดเจน เพื่อให้สามารถทำงานแต่ละขั้นตอนได้สะดวก

#### Acceptance Criteria

1. THE Application SHALL provide three distinct sections: "รวบรวมข้อมูล", "เทรนโมเดล", and "ทำนายผล"
2. THE Application SHALL display a navigation interface using tabs or selectbox for section selection
3. WHEN User selects a section, THE Application SHALL display the corresponding interface for that section

### Requirement 2: การรวบรวมข้อมูล (Data Collection)

**User Story:** ในฐานะผู้ใช้ ฉันต้องการอัปโหลดรูปภาพและบันทึกค่า RGB พร้อมค่าความเข้มข้นจริง เพื่อสร้างชุดข้อมูลสำหรับเทรนโมเดล

#### Acceptance Criteria

1. THE Application SHALL provide a file uploader interface for image upload in the Data Collection section
2. WHEN User uploads an image file, THE Application SHALL display the uploaded image
3. WHEN User uploads an image file, THE Application SHALL calculate average RGB values from a 100x100 pixel region at the center of the image
4. THE Application SHALL display the calculated R, G, B values to the User
5. THE Application SHALL provide a number input field for User to enter the actual Concentration Value
6. THE Application SHALL provide a "บันทึกข้อมูล" button
7. WHEN User clicks the "บันทึกข้อมูล" button, THE Application SHALL append the RGB values and Concentration Value to the Dataset File
8. WHEN User clicks the "บันทึกข้อมูล" button, THE Application SHALL create the Dataset File if it does not exist
9. WHEN data is saved successfully, THE Application SHALL display a success message to the User

### Requirement 3: การเทรนโมเดล (Model Training)

**User Story:** ในฐานะผู้ใช้ ฉันต้องการเทรนโมเดล Machine Learning จากข้อมูลที่รวบรวมไว้ เพื่อใช้ทำนายค่าความเข้มข้นจากภาพใหม่

#### Acceptance Criteria

1. THE Application SHALL provide a "เริ่มเทรนโมเดล" button in the Model Training section
2. WHEN User clicks the "เริ่มเทรนโมเดล" button, THE Application SHALL load data from the Dataset File
3. WHEN User clicks the "เริ่มเทรนโมเดล" button, THE Application SHALL split the data into features (R, G, B) and target (Concentration Value)
4. WHEN User clicks the "เริ่มเทรนโมเดล" button, THE Application SHALL train an ML Model using the loaded data
5. WHEN training is complete, THE Application SHALL calculate and display the R-squared score
6. WHEN training is complete, THE Application SHALL save the trained ML Model to the Model File
7. WHEN training is complete, THE Application SHALL display a success message "เทรนโมเดลสำเร็จ"
8. IF the Dataset File does not exist or is empty, THEN THE Application SHALL display an error message to the User

### Requirement 4: การทำนายผล (Prediction)

**User Story:** ในฐานะผู้ใช้ ฉันต้องการอัปโหลดรูปภาพใหม่และให้โมเดลทำนายค่าความเข้มข้น เพื่อวิเคราะห์ตัวอย่างที่ไม่ทราบค่า

#### Acceptance Criteria

1. THE Application SHALL provide a file uploader interface for new image upload in the Prediction section
2. WHEN User uploads a new image, THE Application SHALL load the trained ML Model from the Model File
3. WHEN User uploads a new image, THE Application SHALL calculate average RGB values from the image using the same method as Data Collection
4. WHEN User uploads a new image, THE Application SHALL use the ML Model to predict the Concentration Value from the RGB values
5. WHEN prediction is complete, THE Application SHALL display the predicted Concentration Value with unit (mg/L)
6. WHEN User uploads a new image, THE Application SHALL display the uploaded image
7. IF the Model File does not exist, THEN THE Application SHALL display an error message instructing User to train a model first

### Requirement 5: การจัดการไฟล์และข้อมูล

**User Story:** ในฐานะผู้ใช้ ฉันต้องการให้แอปพลิเคชันจัดการไฟล์ข้อมูลและโมเดลอย่างถูกต้อง เพื่อให้สามารถใช้งานได้อย่างต่อเนื่อง

#### Acceptance Criteria

1. THE Application SHALL store dataset in CSV format with columns: R, G, B, Concentration
2. THE Application SHALL save the ML Model in joblib format as model.joblib
3. THE Application SHALL handle file I/O operations with appropriate error handling
4. WHEN processing images, THE Application SHALL support common image formats (PNG, JPG, JPEG)

### Requirement 6: การประมวลผลภาพ

**User Story:** ในฐานะผู้ใช้ ฉันต้องการให้แอปพลิเคชันประมวลผลภาพได้อย่างถูกต้อง เพื่อให้ได้ค่า RGB ที่แม่นยำ

#### Acceptance Criteria

1. WHEN calculating RGB values, THE Application SHALL extract a 100x100 pixel region from the center of the image
2. WHEN calculating RGB values, THE Application SHALL compute the mean value for each color channel (R, G, B)
3. IF the image is smaller than 100x100 pixels, THEN THE Application SHALL use the entire image for calculation
4. THE Application SHALL convert images to RGB color space before processing
