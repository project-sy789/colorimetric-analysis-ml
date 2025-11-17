# Implementation Plan

- [x] 1. สร้างโครงสร้างโปรเจกต์และไฟล์พื้นฐาน
  - สร้างไฟล์ `app.py` สำหรับ Streamlit application
  - สร้างไฟล์ `requirements.txt` พร้อม dependencies ที่จำเป็น
  - _Requirements: 1.1, 1.2, 5.1, 5.2, 5.4_

- [x] 2. พัฒนา Image Processor Component
  - [x] 2.1 สร้างฟังก์ชัน `extract_rgb_from_image()` สำหรับคำนวณค่า RGB เฉลี่ย
    - รับ input เป็น image file object และ roi_size
    - อ่านภาพด้วย OpenCV และแปลงจาก BGR เป็น RGB
    - คำนวณจุดกึ่งกลางและตัด ROI ขนาด 100x100 พิกเซล
    - คำนวณค่าเฉลี่ยของแต่ละช่องสี (R, G, B)
    - จัดการกรณีที่ภาพเล็กกว่า ROI size
    - _Requirements: 2.3, 6.1, 6.2, 6.3, 6.4_

- [x] 3. พัฒนา Data Collection Module
  - [x] 3.1 สร้างฟังก์ชัน `save_data_to_csv()` สำหรับบันทึกข้อมูล
    - รับ parameters: R, G, B, concentration, filename
    - ตรวจสอบว่าไฟล์มีอยู่หรือไม่
    - สร้างไฟล์ใหม่พร้อม header ถ้ายังไม่มี
    - Append ข้อมูลใหม่ลงไฟล์ CSV
    - _Requirements: 2.7, 2.8, 5.1_
  
  - [x] 3.2 สร้างฟังก์ชัน `data_collection_module()` สำหรับ UI
    - สร้าง file uploader สำหรับอัปโหลดรูปภาพ
    - แสดงภาพที่อัปโหลด
    - เรียกใช้ `extract_rgb_from_image()` เพื่อคำนวณค่า RGB
    - แสดงค่า R, G, B ที่คำนวณได้
    - สร้าง number input สำหรับกรอกค่าความเข้มข้น
    - สร้างปุ่ม "บันทึกข้อมูล" และเรียกใช้ `save_data_to_csv()`
    - แสดง success message เมื่อบันทึกสำเร็จ
    - จัดการ error cases (invalid image, missing concentration)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

- [x] 4. พัฒนา Model Training Module
  - [x] 4.1 สร้างฟังก์ชัน `train_model()` สำหรับเทรนโมเดล
    - โหลดข้อมูลจาก CSV ด้วย Pandas
    - แยก features (R, G, B) และ target (Concentration)
    - สร้าง RandomForestRegressor model
    - เทรนโมเดลด้วยข้อมูล
    - คำนวณ R² score
    - บันทึกโมเดลเป็นไฟล์ joblib
    - Return model และ R² score
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 5.2_
  
  - [x] 4.2 สร้างฟังก์ชัน `model_training_module()` สำหรับ UI
    - สร้างปุ่ม "เริ่มเทรนโมเดล"
    - เรียกใช้ `train_model()` เมื่อกดปุ่ม
    - แสดง R² score
    - แสดง success message "เทรนโมเดลสำเร็จ"
    - จัดการ error cases (dataset not found, insufficient data)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 5. พัฒนา Prediction Module
  - [x] 5.1 สร้างฟังก์ชัน `load_model()` สำหรับโหลดโมเดล
    - โหลดโมเดลจากไฟล์ joblib
    - จัดการ error ถ้าไม่พบไฟล์
    - Return model object
    - _Requirements: 4.2, 4.7_
  
  - [x] 5.2 สร้างฟังก์ชัน `prediction_module()` สำหรับ UI
    - สร้าง file uploader สำหรับอัปโหลดรูปภาพใหม่ (ใช้ key='predict')
    - แสดงภาพที่อัปโหลด
    - เรียกใช้ `load_model()` เพื่อโหลดโมเดล
    - เรียกใช้ `extract_rgb_from_image()` เพื่อคำนวณค่า RGB
    - ใช้โมเดลทำนายค่าความเข้มข้น
    - แสดงผลลัพธ์พร้อมหน่วย (mg/L)
    - จัดการ error cases (model not found, image processing error)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 6. สร้าง Main Application และ Navigation
  - สร้างฟังก์ชัน `main()` เป็น entry point
  - ตั้งค่า page config และ title
  - สร้าง tabs สำหรับ 3 โมดูล: "รวบรวมข้อมูล", "เทรนโมเดล", "ทำนายผล"
  - เรียกใช้ฟังก์ชันของแต่ละโมดูลใน tab ที่เหมาะสม
  - เพิ่ม `if __name__ == "__main__"` block
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 7. เพิ่ม Error Handling และ Validation
  - เพิ่ม try-except blocks ในทุกฟังก์ชันที่ทำงานกับไฟล์
  - ตรวจสอบ file format ของภาพที่อัปโหลด
  - ตรวจสอบจำนวนข้อมูลก่อนเทรนโมเดล
  - แสดง error messages ที่เป็นมิตรกับผู้ใช้
  - _Requirements: 2.9, 3.8, 4.7, 5.3_

- [x] 8. เขียน Unit Tests
  - เขียน tests สำหรับ `extract_rgb_from_image()` กับภาพขนาดต่างๆ
  - เขียน tests สำหรับ `save_data_to_csv()` และ `train_model()`
  - เขียน tests สำหรับ `load_model()` และการทำนาย
  - _Requirements: All_

- [x] 9. สร้าง Documentation
  - เขียน README.md พร้อมคำแนะนำการติดตั้งและใช้งาน
  - เพิ่ม docstrings ในทุกฟังก์ชัน
  - เพิ่ม comments ในโค้ดที่ซับซ้อน
  - _Requirements: All_
