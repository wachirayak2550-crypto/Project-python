"""
STEP 3: เทรนโมเดลจากข้อมูลที่อัดไว้ + เช็คว่าตัวไหนสับสนกัน

โปรแกรมนี้จะ:
1. อ่านไฟล์ data/<ตัว>.csv ทุกไฟล์ที่มีอยู่จริง (ตัวไหนยังไม่อัดข้อมูล จะข้ามไปเฉยๆ ไม่ error)
2. เอาข้อมูลทั้งหมดมาเทรนโมเดล RandomForest ให้ทายว่า 63 ตัวเลข = ตัวไหน
3. แบ่งข้อมูลส่วนหนึ่งไว้ "ทดสอบ" (ไม่ได้ใช้เทรน) เพื่อดูว่าโมเดลแม่นแค่ไหนกับข้อมูลที่ไม่เคยเห็น
4. แสดง confusion matrix (ตารางสรุปว่าตัวไหนทายถูก/ทายผิดเป็นตัวไหนบ่อย)
5. เซฟโมเดลที่เทรนเสร็จแล้วเป็นไฟล์ model.pkl ไว้ใช้ต่อใน STEP 4

หมายเหตุ: รันไฟล์นี้ใหม่ได้ทุกครั้งที่อัดข้อมูลเพิ่ม (STEP 2) เพื่อให้โมเดลรู้จักตัวใหม่
หรือแม่นขึ้นจากข้อมูลที่เพิ่มเข้ามา
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import database as db

# เตรียมฐานข้อมูลให้พร้อม (สร้างตาราง + ย้ายคำจาก labels.py อัตโนมัติถ้าเป็นเครื่องใหม่)
db.ensure_ready()

# รายชื่อคำที่จะเทรน ตอนนี้ดึงจากฐานข้อมูล (ตาราง labels) แทนไฟล์ labels.py แบบเดิม
LABELS = db.get_label_names()

DATA_DIR = "data"
MODEL_PATH = "model.pkl"


def label_to_filename(label):
    """
    แปลงชื่อคำให้ใช้เป็นชื่อไฟล์ได้ปลอดภัย (ต้องเหมือนกับใน 2_collect_data.py เป๊ะ)
    บางคำมีเครื่องหมาย "/" อยู่ในชื่อ (เช่น "หยุด/อย่า") ต้องแทนที่ด้วย "_" ก่อนใช้เป็นชื่อไฟล์
    """
    return label.replace("/", "_")


def load_all_data():
    """
    อ่านไฟล์ data/<ตัว>.csv ทุกตัวที่มีอยู่จริง (จาก labels.py)
    คืนค่าเป็น X (list ของแถวตัวเลข 63 ค่า) และ y (list ของชื่อตัว ที่ตรงกับแต่ละแถวใน X)
    """
    X = []
    y = []

    for label in LABELS:
        file_path = os.path.join(DATA_DIR, f"{label_to_filename(label)}.csv")
        if not os.path.exists(file_path):
            # ตัวนี้ยังไม่ได้อัดข้อมูล -> ข้ามไป ไม่ error
            continue

        rows = np.loadtxt(file_path, delimiter=",", ndmin=2)
        for row in rows:
            X.append(row)
            y.append(label)

    return np.array(X), np.array(y)


# ---------- โหลดข้อมูลทั้งหมด ----------
X, y = load_all_data()

if len(X) == 0:
    print("ยังไม่มีข้อมูลเลย! ต้องรัน 2_collect_data.py อัดข้อมูลก่อนอย่างน้อย 1 ตัว")
    exit()

# นับจำนวนตัวอย่างของแต่ละตัว โชว์ให้ดูก่อนเทรน
print("จำนวนข้อมูลที่มีตอนนี้:")
unique_labels = sorted(set(y))
for label in unique_labels:
    count = int(np.sum(y == label))
    print(f"  ตัว '{label}': {count} ตัวอย่าง")
    if count < 20:
        print(f"    (แนะนำให้อัดเพิ่ม ตอนนี้มีน้อยไปหน่อย โมเดลอาจไม่แม่น)")

print(f"\nรวมทั้งหมด {len(X)} ตัวอย่าง จาก {len(unique_labels)} ตัว\n")

# ---------- แบ่งข้อมูลเป็น 2 ส่วน: เอาไว้เทรน กับ เอาไว้ทดสอบ ----------
# test_size=0.2 คือ กันข้อมูล 20% ไว้ทดสอบ ไม่เอาไปเทรนด้วย
# stratify=y ช่วยให้แบ่งข้อมูลของแต่ละตัวอย่างสัดส่วนเท่าๆ กัน ไม่ให้ตัวไหนหายไปจากฝั่งทดสอบ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- สร้างและเทรนโมเดล RandomForest ----------
# RandomForest คือการรวมกันของ "ต้นไม้ตัดสินใจ" หลายๆ ต้น มาช่วยกันโหวตทายคำตอบ
# n_estimators=100 คือใช้ต้นไม้ 100 ต้น (ค่ามาตรฐาน เพียงพอสำหรับงานนี้)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------- ทดสอบความแม่นยำกับข้อมูลที่โมเดลไม่เคยเห็น ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"ความแม่นยำของโมเดล (accuracy): {accuracy * 100:.1f}%")
print("(คำนวณจากข้อมูลส่วนที่กันไว้ทดสอบ ไม่ได้ใช้เทรน จึงบอกได้ว่าโมเดลแม่นจริงแค่ไหน)\n")

# ---------- แสดง confusion matrix ----------
# confusion matrix คือตาราง: แถว = คำตอบจริง, คอลัมน์ = คำตอบที่โมเดลทาย
# ถ้าตัวเลขอยู่ "แนวทแยง" มากๆ แปลว่าโมเดลทายถูกบ่อย
# ถ้าตัวเลขอยู่ "นอกแนวทแยง" เยอะ แปลว่าโมเดลสับสนระหว่าง 2 ตัวนั้น
matrix_labels = sorted(set(y))  # เรียงชื่อตัวให้อ่านตารางง่าย
cm = confusion_matrix(y_test, y_pred, labels=matrix_labels)

print("Confusion Matrix (แถว = ตัวจริง, คอลัมน์ = ตัวที่โมเดลทาย):")
header = "        " + "".join(f"{label:>6}" for label in matrix_labels)
print(header)
for true_label, row in zip(matrix_labels, cm):
    row_text = "".join(f"{value:>6}" for value in row)
    print(f"true={true_label:<4}{row_text}")

# ---------- หาคู่ตัวที่โมเดลสับสนกันบ่อยที่สุด มาเตือนผู้ใช้ ----------
print("\nคู่ตัวที่โมเดลสับสนกันบ่อย (ถ้ามี):")
found_confusion = False
for i, true_label in enumerate(matrix_labels):
    for j, pred_label in enumerate(matrix_labels):
        if i == j:
            continue  # ข้ามแนวทแยง (ทายถูก ไม่ใช่ความสับสน)
        mistakes = cm[i][j]
        if mistakes > 0:
            found_confusion = True
            print(f"  จริงคือ '{true_label}' แต่โมเดลทายเป็น '{pred_label}' ผิดไป {mistakes} ครั้ง")

if not found_confusion:
    print("  ไม่มีเลย! โมเดลทายถูกหมดในชุดทดสอบ")
else:
    print("\n  ถ้าคู่ไหนสับสนบ่อยมาก ลองอัดข้อมูลของตัวนั้นเพิ่ม หรือพิจารณาตัดตัวใดตัวหนึ่งออก")

# ---------- เซฟโมเดลลงไฟล์ ----------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nเซฟโมเดลเรียบร้อยแล้วที่ไฟล์ {MODEL_PATH}")
print("ไปทดสอบทายสดได้ที่ STEP 4 (4_predict_live.py)")
