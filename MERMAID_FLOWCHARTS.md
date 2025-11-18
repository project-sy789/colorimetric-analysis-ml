# 🔬 Mermaid Flow Charts สำหรับ https://mermaid.live/

## วิธีใช้งาน:
1. ไปที่ https://mermaid.live/
2. Copy code ด้านล่างทั้งหมด
3. Paste ลงในช่อง Code
4. คลิก "Download PNG" หรือ "Download SVG" เพื่อบันทึกภาพ

---

## Flow Chart 1: ภาพรวมระบบทั้งหมด (System Overview)

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#90EE90','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000','secondaryColor':'#87CEEB','tertiaryColor':'#FFD700'}}}%%
flowchart TB
    Start([🚀 เริ่มต้นใช้งาน]) --> SelectProfile[🧪 เลือก Profile สาร<br/>Phosphate / Nitrate / Ammonia]
    SelectProfile --> Choice{เลือกโหมดการทำงาน}
    
    Choice -->|รวบรวมข้อมูล| Collect[📊 โหมดรวบรวมข้อมูล]
    Choice -->|เทรนโมเดล| Train[🤖 โหมดเทรนโมเดล]
    Choice -->|ทำนายผล| Predict[🔮 โหมดทำนายผล]
    
    %% Data Collection Flow
    Collect --> Upload1[📤 อัปโหลดรูปภาพ<br/>PNG/JPG/JPEG]
    Upload1 --> SetROI1[⚙️ ตั้งค่า ROI<br/>• ขนาด 20-300 px<br/>• ตำแหน่ง X,Y]
    SetROI1 --> ExtractRGB1[🎨 คำนวณค่า RGB เฉลี่ย<br/>จากพื้นที่ ROI]
    ExtractRGB1 --> ShowRGB1[📊 แสดงค่า R, G, B]
    ShowRGB1 --> InputConc[✏️ กรอกค่าความเข้มข้นจริง<br/>หน่วย: mg/L, ppm, etc.]
    InputConc --> SaveData[💾 บันทึกลง CSV<br/>profile_dataset.csv]
    SaveData --> MoreData{ต้องการรวบรวม<br/>ข้อมูลเพิ่มเติม?}
    MoreData -->|ใช่ ต้องการ| Upload1
    MoreData -->|ไม่ เพียงพอแล้ว| End1([✅ จบการรวบรวมข้อมูล])
    
    %% Model Training Flow
    Train --> CheckData{ตรวจสอบข้อมูล<br/>มี ≥ 5 ตัวอย่าง?}
    CheckData -->|ไม่เพียงพอ| Error1[❌ แจ้งเตือน<br/>ข้อมูลไม่เพียงพอ<br/>ต้องมีอย่างน้อย 5 ตัวอย่าง]
    Error1 --> End2([🔴 จบ - ไม่สามารถเทรนได้])
    CheckData -->|เพียงพอ| LoadData[📂 โหลดข้อมูลจาก CSV]
    LoadData --> ValidateData[✔️ ตรวจสอบความถูกต้อง<br/>• คอลัมน์ครบ<br/>• RGB 0-255<br/>• Conc ≥ 0]
    ValidateData --> PrepareData[🔧 เตรียมข้อมูล<br/>X = R, G, B<br/>Y = Concentration]
    PrepareData --> TrainModel[🎓 เทรน Random Forest<br/>n_estimators=100<br/>random_state=42]
    TrainModel --> CalcR2[📈 คำนวณ R² Score<br/>วัดความแม่นยำ]
    CalcR2 --> SaveModel[💾 บันทึก Model<br/>profile_model.joblib]
    SaveModel --> CalcLOD[� คำ นวณ LOD<br/>จาก blank samples<br/>LOD = mean + 3×SD]
    CalcLOD --> PlotCurve[📈 สร้าง Calibration Curve<br/>กราฟ Actual vs Predicted<br/>แสดง R² score]
    PlotCurve --> ShowResult[📊 แสดงผล<br/>• R² Score<br/>• LOD value<br/>• Calibration Curve]
    ShowResult --> End3([✅ จบการเทรนโมเดล])
    
    %% Prediction Flow
    Predict --> CheckModel{ตรวจสอบ<br/>มี Model ที่เทรนแล้ว?}
    CheckModel -->|ไม่มี| Error2[❌ แจ้งเตือน<br/>กรุณาเทรนโมเดลก่อน]
    Error2 --> End4([🔴 จบ - ไม่สามารถทำนายได้])
    CheckModel -->|มี| Upload2[📤 อัปโหลดรูปภาพใหม่]
    Upload2 --> SetROI2[⚙️ ตั้งค่า ROI<br/>ใช้การตั้งค่าเดียวกับตอนเทรน]
    SetROI2 --> ExtractRGB2[🎨 คำนวณค่า RGB เฉลี่ย<br/>จากพื้นที่ ROI]
    ExtractRGB2 --> ShowRGB2[📊 แสดงค่า R, G, B]
    ShowRGB2 --> LoadModel[📂 โหลด Model]
    LoadModel --> PredictConc[🔮 ทำนายค่าความเข้มข้น<br/>จาก RGB values]
    PredictConc --> ShowPrediction[🎯 แสดงผลการทำนาย<br/>พร้อมหน่วย]
    ShowPrediction --> End5([✅ จบการทำนาย])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef collectStyle fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef trainStyle fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef predictStyle fill:#DDA0DD,stroke:#000,stroke-width:2px,color:#000
    classDef errorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    
    class Start startStyle
    class End1,End2,End3,End4,End5 endStyle
    class Collect,Upload1,SetROI1,ExtractRGB1,ShowRGB1,InputConc,SaveData collectStyle
    class Train,CheckData,LoadData,ValidateData,PrepareData,TrainModel,CalcR2,SaveModel,ShowResult trainStyle
    class Predict,Upload2,SetROI2,ExtractRGB2,ShowRGB2,LoadModel,PredictConc,ShowPrediction predictStyle
    class Error1,Error2 errorStyle

---

## Flow Chart 2: กระบวนการประมวลผลภาพ (Image Processing)

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#87CEEB','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000'}}}%%
flowchart TB
    Start([📸 รับภาพจาก User]) --> Validate{ตรวจสอบรูปแบบไฟล์}
    Validate -->|ไม่ถูกต้อง| Error[❌ Error<br/>รองรับเฉพาะ PNG/JPG/JPEG]
    Validate -->|ถูกต้อง| ReadImage[📂 อ่านภาพด้วย OpenCV<br/>cv2.imdecode]
    Error --> EndError([🔴 จบ - ไม่สามารถประมวลผล])
    
    ReadImage --> CheckEmpty{ตรวจสอบ<br/>ภาพว่างเปล่า?}
    CheckEmpty -->|ว่างเปล่า| Error2[❌ Error<br/>ไฟล์ภาพว่างเปล่า]
    Error2 --> EndError
    CheckEmpty -->|ไม่ว่าง| ConvertColor[🎨 แปลงสี<br/>BGR → RGB<br/>cv2.cvtColor]
    
    ConvertColor --> GetSize[📏 หาขนาดภาพ<br/>Height × Width]
    GetSize --> CalcCenter[🎯 คำนวณจุดกลางภาพ<br/>Center_X = Width ÷ 2<br/>Center_Y = Height ÷ 2]
    CalcCenter --> ApplyOffset[⚙️ ปรับตำแหน่ง ROI<br/>ROI_X = Center_X + Offset_X<br/>ROI_Y = Center_Y + Offset_Y]
    ApplyOffset --> CheckSize{ภาพใหญ่กว่า<br/>ROI Size?}
    
    CheckSize -->|ไม่ใหญ่กว่า| UseFullImage[📐 ใช้ภาพทั้งหมด<br/>ROI = ภาพเต็ม]
    CheckSize -->|ใหญ่กว่า| ExtractROI[✂️ ตัดเอาพื้นที่ ROI<br/>ROI_Size × ROI_Size pixels]
    
    UseFullImage --> CalcMean[🧮 คำนวณค่าเฉลี่ย<br/>R = mean ROI&#91;:,:,0&#93;<br/>G = mean ROI&#91;:,:,1&#93;<br/>B = mean ROI&#91;:,:,2&#93;]
    ExtractROI --> CalcMean
    
    CalcMean --> ValidateRGB{ตรวจสอบ<br/>ค่า RGB ถูกต้อง?}
    ValidateRGB -->|ไม่ถูกต้อง| Error3[❌ Error<br/>ค่า RGB ไม่ถูกต้อง]
    Error3 --> EndError
    ValidateRGB -->|ถูกต้อง| DrawBox[🖼️ วาดกรอบ ROI<br/>สีเขียว + ข้อความ<br/>cv2.rectangle + cv2.putText]
    
    DrawBox --> Return([✅ ส่งคืน<br/>R, G, B values<br/>และภาพที่มีกรอบ ROI])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef processStyle fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef errorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    
    class Start startStyle
    class Return endStyle
    class EndError errorStyle
    class ReadImage,ConvertColor,GetSize,CalcCenter,ApplyOffset,UseFullImage,ExtractROI,CalcMean,DrawBox processStyle
    class Error,Error2,Error3 errorStyle

---

## Flow Chart 3: กระบวนการ Machine Learning

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#FFD700','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000'}}}%%
flowchart TB
    Start([🤖 เริ่มเทรนโมเดล]) --> LoadCSV[📂 โหลดข้อมูล<br/>pd.read_csv<br/>profile_dataset.csv]
    
    LoadCSV --> ValidateData{ตรวจสอบข้อมูล}
    ValidateData -->|ขาดคอลัมน์| Error1[❌ Error<br/>ไฟล์ข้อมูลขาดคอลัมน์<br/>R, G, B, Concentration]
    ValidateData -->|< 5 ตัวอย่าง| Error2[❌ Error<br/>ข้อมูลไม่เพียงพอ<br/>ต้องมีอย่างน้อย 5 ตัวอย่าง]
    ValidateData -->|RGB นอกช่วง 0-255| Error3[❌ Error<br/>ค่า RGB ไม่ถูกต้อง<br/>ต้องอยู่ในช่วง 0-255]
    ValidateData -->|Conc < 0| Error4[❌ Error<br/>ค่าความเข้มข้นติดลบ<br/>ต้อง ≥ 0]
    ValidateData -->|ผ่านทุกเงื่อนไข| SplitData[🔧 แยกข้อมูล<br/>X = df&#91;R, G, B&#93;<br/>Y = df&#91;Concentration&#93;]
    
    Error1 --> End1([🔴 จบ - ไม่สามารถเทรนได้])
    Error2 --> End1
    Error3 --> End1
    Error4 --> End1
    
    SplitData --> CreateModel[🏗️ สร้าง Random Forest Model<br/>RandomForestRegressor<br/>n_estimators = 100<br/>random_state = 42]
    CreateModel --> FitModel[🎓 เทรนโมเดล<br/>model.fit X, Y<br/>เรียนรู้ความสัมพันธ์<br/>RGB → Concentration]
    FitModel --> CalcR2[📊 คำนวณ R² Score<br/>score = model.score X, Y<br/>วัดความแม่นยำ 0-1]
    CalcR2 --> ValidateR2{ตรวจสอบ<br/>R² Score}
    ValidateR2 -->|NaN หรือ Inf| Error5[❌ Error<br/>ไม่สามารถคำนวณ R² ได้]
    Error5 --> End1
    ValidateR2 -->|ค่าปกติ| SaveModel[💾 บันทึก Model<br/>joblib.dump<br/>profile_model.joblib]
    
    SaveModel --> CalcLOD2[🔬 คำนวณ LOD]
    CalcLOD2 --> CheckBlank{มี blank<br/>≥ 3 ตัวอย่าง?}
    CheckBlank -->|ไม่มี| SkipLOD[⚠️ ข้ามการคำนวณ LOD<br/>ต้องมี blank ≥ 3 ตัวอย่าง]
    CheckBlank -->|มี| ComputeLOD[📊 คำนวณ LOD<br/>LOD = mean + 3×SD<br/>จาก blank predictions]
    
    SkipLOD --> PlotCurve2[📈 สร้าง Calibration Curve]
    ComputeLOD --> PlotCurve2
    
    PlotCurve2 --> CreateGraph[🎨 สร้างกราฟ<br/>• Scatter plot Actual vs Predicted<br/>• เส้น ideal y=x<br/>• แสดง R² score<br/>• แสดงจำนวนตัวอย่าง]
    
    CreateGraph --> InterpretR2{ตีความ R² Score}
    
    InterpretR2 -->|R² ≥ 0.9| Excellent[🎯 ความแม่นยำสูงมาก<br/>Excellent<br/>โมเดลพร้อมใช้งาน]
    InterpretR2 -->|0.7 ≤ R² < 0.9| Good[👍 ความแม่นยำดี<br/>Good<br/>โมเดลใช้งานได้]
    InterpretR2 -->|0.5 ≤ R² < 0.7| Moderate[⚠️ ความแม่นยำปานกลาง<br/>Moderate<br/>ควรรวบรวมข้อมูลเพิ่ม]
    InterpretR2 -->|R² < 0.5| Poor[❌ ความแม่นยำต่ำ<br/>Poor<br/>ต้องรวบรวมข้อมูลเพิ่ม]
    
    Excellent --> ShowMetrics[📈 แสดงผลลัพธ์<br/>• R² Score<br/>• ความแม่นยำ %<br/>• LOD value<br/>• Calibration Curve<br/>• จำนวนข้อมูล]
    Good --> ShowMetrics
    Moderate --> ShowMetrics
    Poor --> ShowMetrics
    
    ShowMetrics --> End2([✅ จบการเทรนโมเดล<br/>Model พร้อมใช้งาน])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef endErrorStyle fill:#FF6B6B,stroke:#000,stroke-width:3px,color:#fff
    classDef processStyle fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef errorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    classDef excellentStyle fill:#90EE90,stroke:#000,stroke-width:2px,color:#000
    classDef goodStyle fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef moderateStyle fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef poorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    
    class Start startStyle
    class End2 endStyle
    class End1 endErrorStyle
    class LoadCSV,SplitData,CreateModel,FitModel,CalcR2,SaveModel,ShowMetrics processStyle
    class Error1,Error2,Error3,Error4,Error5 errorStyle
    class Excellent excellentStyle
    class Good goodStyle
    class Moderate moderateStyle
    class Poor poorStyle

---

## Flow Chart 4: ขั้นตอนการใช้งานทั้งหมด (User Journey)

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#DDA0DD','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000'}}}%%
flowchart TB
    Start([👤 ผู้ใช้เริ่มต้น]) --> OpenApp[🌐 เปิดแอปพลิเคชัน<br/>http://localhost:8501]
    OpenApp --> SelectProfile[🧪 เลือก/สร้าง Profile<br/>• เลือกสารที่ต้องการวิเคราะห์<br/>• หรือเพิ่ม Profile ใหม่<br/>• ตั้งหน่วยความเข้มข้น]
    
    SelectProfile --> Phase1{มีข้อมูล<br/>เทรนแล้ว?}
    Phase1 -->|ไม่มี ต้องเริ่มใหม่| Step1[📊 ขั้นตอนที่ 1<br/>รวบรวมข้อมูล]
    Phase1 -->|มีแล้ว| Phase2{มี Model<br/>เทรนแล้ว?}
    
    Step1 --> Prepare[🧪 เตรียมตัวอย่าง<br/>• สารละลายมาตรฐาน<br/>• หลายความเข้มข้น<br/>• เติมสารทำปฏิกิริยา]
    Prepare --> TakePhoto1[📸 ถ่ายภาพ<br/>• ใช้กล่องแสง<br/>• การตั้งค่าเดียวกัน<br/>• ถ่ายทุกตัวอย่าง]
    TakePhoto1 --> Upload1[📤 อัปโหลดภาพ<br/>ในแอปพลิเคชัน]
    Upload1 --> SetROI1[⚙️ ตั้งค่า ROI<br/>• ปรับขนาด<br/>• ปรับตำแหน่ง<br/>• ดูกรอบสีเขียว]
    SetROI1 --> Input1[✏️ กรอกค่าความเข้มข้นจริง<br/>ที่วัดได้]
    Input1 --> Save1[💾 บันทึกข้อมูล]
    Save1 --> More1{มีตัวอย่าง<br/>เพิ่มเติม?}
    More1 -->|ใช่| TakePhoto1
    More1 -->|ไม่ ครบแล้ว| Check1{มีข้อมูล<br/>≥ 10 ตัวอย่าง?}
    Check1 -->|ไม่ ควรเพิ่ม| Recommend1[💡 แนะนำ<br/>ควรมี 10-20 ตัวอย่าง<br/>เพื่อความแม่นยำ]
    Recommend1 --> More1
    Check1 -->|ใช่ เพียงพอ| Step2[🤖 ขั้นตอนที่ 2<br/>เทรนโมเดล]
    
    Phase2 -->|ไม่มี| Step2
    Phase2 -->|มีแล้ว| Step3[🔮 ขั้นตอนที่ 3<br/>ทำนายผล]
    
    Step2 --> Train1[🎓 คลิกเทรนโมเดล]
    Train1 --> Wait1[⏳ รอการเทรน<br/>ไม่กี่วินาที]
    Wait1 --> ShowR2[📊 ดู R² Score]
    ShowR2 --> CheckR2{R² Score<br/>≥ 0.7?}
    CheckR2 -->|ไม่ ต่ำเกินไป| Recommend2[💡 แนะนำ<br/>ควรรวบรวมข้อมูลเพิ่ม<br/>หรือตรวจสอบคุณภาพข้อมูล]
    Recommend2 --> Choice1{ต้องการ<br/>ทำอะไร?}
    Choice1 -->|เพิ่มข้อมูล| Step1
    Choice1 -->|ใช้ Model นี้ต่อ| Step3
    CheckR2 -->|ใช่ ดี| Step3
    
    Step3 --> TakePhoto2[📸 ถ่ายภาพตัวอย่างใหม่<br/>ที่ไม่รู้ความเข้มข้น]
    TakePhoto2 --> Upload2[📤 อัปโหลดภาพ]
    Upload2 --> SetROI2[⚙️ ตั้งค่า ROI<br/>ใช้การตั้งค่าเดียวกับตอนเทรน]
    SetROI2 --> Predict1[🔮 ดูผลการทำนาย]
    Predict1 --> ShowResult[🎯 แสดงค่าความเข้มข้น<br/>พร้อมหน่วย]
    ShowResult --> Verify{ต้องการ<br/>ตรวจสอบความถูกต้อง?}
    Verify -->|ใช่| Compare[📊 เปรียบเทียบกับค่าจริง<br/>ถ้ามี]
    Compare --> Accurate{ผลลัพธ์<br/>แม่นยำ?}
    Accurate -->|ไม่แม่นยำ| Recommend3[💡 แนะนำ<br/>• ตรวจสอบการตั้งค่า ROI<br/>• เพิ่มข้อมูลเทรน<br/>• เทรนโมเดลใหม่]
    Recommend3 --> Choice2{ต้องการ<br/>ทำอะไร?}
    Choice2 -->|เพิ่มข้อมูล| Step1
    Choice2 -->|เทรนใหม่| Step2
    Choice2 -->|ทำนายต่อ| More2
    Accurate -->|แม่นยำ| More2{มีตัวอย่าง<br/>อื่นที่ต้องการทำนาย?}
    Verify -->|ไม่| More2
    More2 -->|ใช่| TakePhoto2
    More2 -->|ไม่| End([✅ เสร็จสิ้น<br/>ปิดแอปพลิเคชัน])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef step1Style fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef step2Style fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef step3Style fill:#DDA0DD,stroke:#000,stroke-width:2px,color:#000
    classDef recommendStyle fill:#FFA500,stroke:#000,stroke-width:2px,color:#000
    
    class Start startStyle
    class End endStyle
    class Step1,Prepare,TakePhoto1,Upload1,SetROI1,Input1,Save1 step1Style
    class Step2,Train1,Wait1,ShowR2 step2Style
    class Step3,TakePhoto2,Upload2,SetROI2,Predict1,ShowResult step3Style
    class Recommend1,Recommend2,Recommend3 recommendStyle

---

## คำแนะนำการใช้งาน:

### สำหรับ https://mermaid.live/
1. Copy code ทั้งหมดของ Flow Chart ที่ต้องการ (รวม ```mermaid และ ```)
2. Paste ลงในช่อง "Code" ทางซ้าย
3. ดูผลลัพธ์ทางขวา
4. ปรับแต่งสีหรือรูปแบบได้ที่บรรทัด `%%{init:...}%%`
5. Download เป็น PNG หรือ SVG

### การปรับแต่งสี:
- `primaryColor`: สีหลัก
- `primaryTextColor`: สีข้อความ
- `primaryBorderColor`: สีขอบ
- `lineColor`: สีเส้นเชื่อม

### ขนาดภาพที่แนะนำ:
- **สำหรับสไลด์**: 1920x1080 px (16:9)
- **สำหรับเอกสาร**: 1200x800 px
- **สำหรับโปสเตอร์**: 2400x1600 px

### Tips:
- ใช้ Flow Chart 1 สำหรับภาพรวมทั้งหมด
- ใช้ Flow Chart 2 สำหรับอธิบายเทคนิค
- ใช้ Flow Chart 3 สำหรับอธิบาย ML
- ใช้ Flow Chart 4 สำหรับคู่มือผู้ใช้

---

**หมายเหตุ**: Flow Charts เหล่านี้ออกแบบมาเพื่อความชัดเจนและเหมาะสมกับการนำเสนอระดับมัธยมศึกษาตอนปลาย


---

## 🆕 Flow Chart 5: Machine Learning with Feature Importance & Hyperparameter Tuning

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#FFD700','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000'}}}%%
flowchart TB
    Start([🤖 เริ่มเทรนโมเดล]) --> LoadCSV[📂 โหลดข้อมูล<br/>profile_dataset.csv]
    LoadCSV --> ValidateData{ตรวจสอบข้อมูล}
    ValidateData -->|ผ่าน| SplitData[🔧 แยกข้อมูล<br/>X = R,G,B<br/>Y = Concentration]
    ValidateData -->|ไม่ผ่าน| Error[❌ Error]
    Error --> End1([🔴 จบ])
    
    SplitData --> TrainNormal[🎓 เทรนโมเดลปกติ<br/>Random Forest<br/>n_estimators=100]
    TrainNormal --> CalcR2Normal[📊 คำนวณ R² Score]
    CalcR2Normal --> SaveNormal[💾 บันทึก<br/>profile_model.joblib]
    
    SaveNormal --> FeatureImp[🎯 คำนวณ Feature Importance<br/>ความสำคัญของ R, G, B]
    FeatureImp --> ShowFeatureImp[📊 แสดงตารางและกราฟ<br/>R: 45.2%<br/>G: 32.8%<br/>B: 22.0%]
    
    ShowFeatureImp --> CalcLOD[🔬 คำนวณ LOD]
    CalcLOD --> PlotCurve[📈 สร้าง Calibration Curve]
    PlotCurve --> ShowNormalResult[📊 แสดงผลโมเดลปกติ<br/>R² Score, LOD, Curve]
    
    ShowNormalResult --> AskTuning{ต้องการ<br/>Hyperparameter Tuning?}
    AskTuning -->|ไม่| End2([✅ จบ - ใช้โมเดลปกติ])
    AskTuning -->|ใช่| GridSearch[🔧 GridSearchCV<br/>ทดสอบพารามิเตอร์หลายชุด<br/>5-fold cross-validation]
    
    GridSearch --> TestParams[🧪 ทดสอบ<br/>n_estimators: 50-200<br/>max_depth: None-30<br/>min_samples_split: 2-10]
    TestParams --> FindBest[🎯 หาพารามิเตอร์ที่ดีที่สุด]
    FindBest --> TrainTuned[🎓 เทรนโมเดล Tuned<br/>ด้วย Best Parameters]
    
    TrainTuned --> CalcR2Tuned[📊 คำนวณ R² Score<br/>ของโมเดล Tuned]
    CalcR2Tuned --> SaveTuned[💾 บันทึก<br/>profile_model_tuned.joblib]
    SaveTuned --> Compare[📊 เปรียบเทียบ<br/>โมเดลปกติ vs Tuned]
    
    Compare --> ShowComparison[📈 แสดงผล<br/>Normal: R²=98.56%<br/>Tuned: R²=99.12%<br/>Improvement: +0.56%]
    ShowComparison --> ShowBestParams[🎯 แสดง Best Parameters<br/>n_estimators: 200<br/>max_depth: 20<br/>min_samples_split: 2]
    ShowBestParams --> End3([✅ จบ - มีทั้ง 2 โมเดล])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef processStyle fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef featureStyle fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef tuningStyle fill:#DDA0DD,stroke:#000,stroke-width:2px,color:#000
    classDef errorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    
    class Start startStyle
    class End1,End2,End3 endStyle
    class TrainNormal,CalcR2Normal,SaveNormal,CalcLOD,PlotCurve,ShowNormalResult processStyle
    class FeatureImp,ShowFeatureImp featureStyle
    class GridSearch,TestParams,FindBest,TrainTuned,CalcR2Tuned,SaveTuned,Compare,ShowComparison,ShowBestParams tuningStyle
    class Error errorStyle

---

## 🆕 Flow Chart 6: Prediction with Model Selection

%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#DDA0DD','primaryTextColor':'#000','primaryBorderColor':'#000','lineColor':'#000'}}}%%
flowchart TB
    Start([🔮 เริ่มทำนาย]) --> CheckModels{ตรวจสอบโมเดล}
    CheckModels -->|ไม่มี| Error[❌ กรุณาเทรนโมเดลก่อน]
    Error --> End1([🔴 จบ])
    CheckModels -->|มี| Upload[📤 อัปโหลดภาพ]
    
    Upload --> SetROI[⚙️ ตั้งค่า ROI]
    SetROI --> ExtractRGB[🎨 คำนวณ RGB]
    ExtractRGB --> ShowRGB[📊 แสดง R, G, B]
    
    ShowRGB --> SelectModel{เลือกโมเดล}
    SelectModel -->|โมเดลปกติ| LoadNormal[📂 โหลด<br/>profile_model.joblib]
    SelectModel -->|โมเดล Tuned| CheckTuned{มีโมเดล<br/>Tuned?}
    
    CheckTuned -->|ไม่มี| Warning[⚠️ ยังไม่ได้ Tuning<br/>ใช้โมเดลปกติแทน]
    Warning --> LoadNormal
    CheckTuned -->|มี| LoadTuned[📂 โหลด<br/>profile_model_tuned.joblib]
    
    LoadNormal --> PredictNormal[🔮 ทำนาย<br/>ด้วยโมเดลปกติ]
    LoadTuned --> PredictTuned[🔮 ทำนาย<br/>ด้วยโมเดล Tuned]
    
    PredictNormal --> ShowResultNormal[🎯 แสดงผล<br/>ความเข้มข้น<br/>โมเดล: ปกติ<br/>R²: 98.56%]
    PredictTuned --> ShowResultTuned[🎯 แสดงผล<br/>ความเข้มข้น<br/>โมเดล: Tuned<br/>R²: 99.12%]
    
    ShowResultNormal --> More{ทำนายต่อ?}
    ShowResultTuned --> More
    More -->|ใช่| Upload
    More -->|ไม่| End2([✅ จบ])
    
    %% Styling
    classDef startStyle fill:#90EE90,stroke:#000,stroke-width:3px,color:#000
    classDef endStyle fill:#FFB6C1,stroke:#000,stroke-width:3px,color:#000
    classDef processStyle fill:#87CEEB,stroke:#000,stroke-width:2px,color:#000
    classDef normalStyle fill:#FFD700,stroke:#000,stroke-width:2px,color:#000
    classDef tunedStyle fill:#DDA0DD,stroke:#000,stroke-width:2px,color:#000
    classDef errorStyle fill:#FF6B6B,stroke:#000,stroke-width:2px,color:#fff
    
    class Start startStyle
    class End1,End2 endStyle
    class Upload,SetROI,ExtractRGB,ShowRGB processStyle
    class LoadNormal,PredictNormal,ShowResultNormal normalStyle
    class LoadTuned,PredictTuned,ShowResultTuned tunedStyle
    class Error,Warning errorStyle

---

## 📝 หมายเหตุสำหรับ Flow Charts ใหม่

### Flow Chart 5: Machine Learning with Feature Importance & Hyperparameter Tuning
- แสดงขั้นตอนการเทรนโมเดลแบบสมบูรณ์
- รวม Feature Importance Analysis
- รวม Hyperparameter Tuning ด้วย GridSearchCV
- เปรียบเทียบโมเดลปกติกับโมเดล Tuned

### Flow Chart 6: Prediction with Model Selection
- แสดงการเลือกโมเดลในการทำนาย
- รองรับทั้งโมเดลปกติและโมเดล Tuned
- แสดง R² Score ของแต่ละโมเดล
- จัดการกรณีที่ยังไม่มีโมเดล Tuned

### การใช้งาน:
1. Flow Chart 5 ใช้แทน Flow Chart 3 เดิม (เพิ่มฟีเจอร์ใหม่)
2. Flow Chart 6 ใช้เสริม Flow Chart 1 ในส่วน Prediction
3. Flow Chart 1-4 เดิมยังใช้ได้ แต่ไม่มีฟีเจอร์ใหม่

**แนะนำ:** ใช้ Flow Chart 5 และ 6 สำหรับการนำเสนอที่ต้องการแสดงฟีเจอร์ใหม่ทั้งหมด
