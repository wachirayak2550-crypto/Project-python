"""
STEP 2: อัดข้อมูลท่ามือ (เก็บไว้สอนโมเดลใน STEP 3)

วิธีทำงานคร่าวๆ:
1. โปรแกรมถามว่าจะอัดท่าของ "ตัว" ไหน (เลือกจากรายชื่อใน labels.py)
2. เปิดกล้อง ให้ผู้ใช้ทำท่ามือค้างไว้หน้ากล้อง
3. ทุกครั้งที่กดปุ่ม SPACE โปรแกรมจะบันทึก "จุดมือ 63 ตัวเลข" ของเฟรมนั้น
   ลงไฟล์ data/<ตัว>.csv เพิ่มไปเรื่อยๆ ทีละแถว
4. ทำซ้ำแบบนี้ ~50-100 ครั้งต่อตัว (ขยับมือ/มุมกล้องนิดหน่อยระหว่างกด จะช่วยให้โมเดลแม่นขึ้น)

หมายเหตุเรื่อง 2 มือ:
- หน้าจอจะวาดจุดให้เห็นได้ทั้ง 2 มือ (เผื่อยกมือผิดข้าง จะได้เห็น)
- แต่ตอนบันทึกข้อมูลจริง จะ "เลือกมือที่ใหญ่ที่สุดในเฟรม" (มือที่เด่นสุด) มาบันทึกแค่มือเดียว
  เพื่อให้ข้อมูลทุกแถวมีจำนวนตัวเลขเท่ากันเสมอ (63 ตัว) โมเดลจะได้ไม่งง

วิธีใช้: รัน python 2_collect_data.py แล้วทำตามคำแนะนำในโปรแกรม
กด SPACE เพื่อบันทึกท่าปัจจุบัน, กด BACKSPACE เพื่อลบท่าล่าสุดที่เพิ่งบันทึกไป (เผื่อกดพลาด),
กด 'q' เพื่อออกจากโปรแกรม
"""

import os
import csv

import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# เช็คก่อนว่า mediapipe เวอร์ชันนี้มีฟีเจอร์จับมือที่เราต้องใช้ไหม
# (ถ้าเครื่องใช้ Python ใหม่เกินไป เช่น 3.12/3.13 อาจได้ mediapipe เวอร์ชันที่ตัดฟีเจอร์นี้ทิ้งไปแล้ว)
if not hasattr(mp, "solutions"):
    print("เจอปัญหา: mediapipe เวอร์ชันนี้ไม่มีฟีเจอร์จับมือที่โปรแกรมนี้ต้องใช้")
    print("สาเหตุที่พบบ่อยที่สุด: เครื่องนี้ใช้ Python เวอร์ชันใหม่เกินไป (เช่น 3.12 หรือ 3.13)")
    print("")
    print("วิธีแก้:")
    print("1. ติดตั้ง Python 3.11 จาก https://www.python.org/downloads/release/python-3119/")
    print("2. ติดตั้ง library ใหม่ด้วย Python 3.11:  py -3.11 -m pip install -r requirements.txt")
    print("3. รันไฟล์นี้ใหม่ด้วย Python 3.11:  py -3.11 2_collect_data.py")
    exit()

import database as db

# รายชื่อคำที่จะอัด ตอนนี้ดึงจากฐานข้อมูล (ตาราง labels) แทนไฟล์ labels.py แบบเดิม
# เพราะแอดมินสามารถเพิ่ม/ลบคำผ่านหน้าเว็บได้แล้ว ฐานข้อมูลจึงเป็นแหล่งข้อมูลหลักตอนนี้
LABELS = db.get_label_names()

# cv2.putText ธรรมดา "วาดภาษาไทยไม่ได้" (จะขึ้นเป็น ??? แทน)
# เลยต้องใช้ Pillow (PIL) ช่วยวาดตัวหนังสือไทยแทน โดยใช้ฟอนต์ Tahoma ที่มากับ Windows
THAI_FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"
thai_font = ImageFont.truetype(THAI_FONT_PATH, 28)


def draw_thai_text(frame, text, position, font=thai_font, color_bgr=(0, 255, 0)):
    """
    วาดข้อความภาษาไทย (หรือข้อความอะไรก็ได้) ลงบนภาพ frame
    วิธีทำ: แปลงภาพเป็นรูปแบบของ Pillow -> วาดตัวหนังสือ -> แปลงกลับเป็นภาพของ OpenCV
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # PIL ใช้สีแบบ RGB ไม่ใช่ BGR
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- เตรียมตัวจับมือของ MediaPipe (เหมือน STEP 1) ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ---------- เตรียมโฟลเดอร์เก็บข้อมูล ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def hand_bbox_area(hand_landmarks):
    """
    คำนวณ "พื้นที่กรอบสี่เหลี่ยม" ที่ล้อมรอบมือ 1 ข้าง
    ใช้พิกัด x, y ของจุดทั้ง 21 จุด (เป็นค่า 0.0-1.0 เทียบกับขนาดภาพ)
    มือที่อยู่ใกล้กล้อง/เด่นกว่า จะมีพื้นที่กรอบนี้ใหญ่กว่ามือที่อยู่ไกลๆ หรือเล็กๆ
    เราใช้ค่านี้เพื่อเลือกว่า "มือไหนเด่นสุดในเฟรม"
    """
    xs = [point.x for point in hand_landmarks.landmark]
    ys = [point.y for point in hand_landmarks.landmark]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return width * height


def landmarks_to_list(hand_landmarks):
    """
    แปลงจุดมือ 21 จุดของ MediaPipe ให้เป็น list ตัวเลขธรรมดา
    แต่ละจุดมี x, y, z ดังนั้น 21 จุด x 3 ค่า = 63 ตัวเลข
    ลำดับตัวเลขจะเหมือนกันเป๊ะทุกครั้ง เพราะ MediaPipe เรียงจุดตามลำดับคงที่เสมอ
    """
    row = []
    for point in hand_landmarks.landmark:
        row.append(point.x)
        row.append(point.y)
        row.append(point.z)
    return row


def label_to_filename(label):
    """
    แปลงชื่อคำให้ใช้เป็นชื่อไฟล์ได้ปลอดภัย
    บางคำมีเครื่องหมาย "/" อยู่ในชื่อ (เช่น "หยุด/อย่า") ซึ่งคอมพิวเตอร์จะเข้าใจผิด
    ว่าเป็นการแบ่งโฟลเดอร์ย่อย เลยต้องแทนที่ "/" ด้วย "_" ก่อนใช้เป็นชื่อไฟล์
    """
    return label.replace("/", "_")


def count_saved_rows(label):
    """นับว่าไฟล์ของตัวนี้มีข้อมูลบันทึกไว้แล้วกี่แถว (กี่ตัวอย่าง)"""
    file_path = os.path.join(DATA_DIR, f"{label_to_filename(label)}.csv")
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", newline="") as f:
        return sum(1 for _ in csv.reader(f))


# ---------- ให้ผู้ใช้เลือกว่าจะอัดตัวไหน ----------
print("มีตัวที่ต้องอัดข้อมูลทั้งหมดดังนี้:")
for i, label in enumerate(LABELS):
    already = count_saved_rows(label)
    print(f"  [{i}] {label}  (อัดไปแล้ว {already} ครั้ง)")

choice = input("พิมพ์หมายเลขของตัวที่จะอัดข้อมูลตอนนี้: ")
chosen_index = int(choice)
current_label = LABELS[chosen_index]

output_path = os.path.join(DATA_DIR, f"{label_to_filename(current_label)}.csv")
print(f"\nกำลังอัดข้อมูลของตัว '{current_label}' -> จะบันทึกลงไฟล์ {output_path}")
print("ทำท่ามือค้างไว้หน้ากล้อง แล้วกด SPACE เพื่อบันทึก 1 ครั้ง")
print("แนะนำให้ขยับมือ/มุมนิดหน่อยระหว่างกด จะได้ข้อมูลหลากหลาย (เป้าหมาย ~50-100 ครั้ง)")
print("กด 'q' เมื่ออัดพอแล้วเพื่อออกจากโปรแกรม\n")

# ---------- เปิดกล้อง ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("เปิดกล้องไม่ได้ ลองเช็คว่ามีโปรแกรมอื่นใช้กล้องอยู่หรือเปล่า")
    exit()

saved_count = count_saved_rows(current_label)

# ตำแหน่ง (byte) ของไฟล์ก่อนจะบันทึกท่าล่าสุด ใช้สำหรับ "ลบท่าล่าสุด" ด้วย BACKSPACE
# ถ้ายังไม่เคยบันทึกอะไรในรอบนี้เลย ค่านี้จะเป็น None (แปลว่ายังลบอะไรไม่ได้)
last_save_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("อ่านภาพจากกล้องไม่ได้ หยุดโปรแกรม")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # dominant_hand คือมือที่เด่นสุดในเฟรมนี้ (ใช้บันทึกข้อมูล)
    # ถ้าไม่เจอมือเลย ค่านี้จะเป็น None
    dominant_hand = None
    best_area = -1

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # วาดจุดทุกมือที่เจอ ให้เห็นบนจอ (เพื่อ debug/ดูสมบูรณ์)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # เช็คว่ามือนี้ใหญ่กว่ามือก่อนหน้าไหม ถ้าใช่ให้เป็นตัวเก็บไว้ใช้บันทึก
            area = hand_bbox_area(hand_landmarks)
            if area > best_area:
                best_area = area
                dominant_hand = hand_landmarks

    # ---------- แสดงข้อความช่วยเหลือบนจอ (ใช้ draw_thai_text เพราะมีตัวหนังสือไทย) ----------
    frame = draw_thai_text(frame, f"Label: {current_label}  Saved: {saved_count}", (10, 10))
    frame = draw_thai_text(frame, "SPACE = save, BACKSPACE = undo last, q = quit", (10, 45))

    if dominant_hand is None:
        frame = draw_thai_text(frame, "ไม่เจอมือ - ยกมือเข้ามาในเฟรม", (10, 80), color_bgr=(0, 0, 255))

    cv2.imshow("HandSpeak - Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # 32 คือรหัสของปุ่ม SPACE
        if dominant_hand is None:
            print("ยังไม่เจอมือในเฟรม ยกมือเข้ามาก่อนแล้วค่อยกด SPACE")
        else:
            # จำตำแหน่งไฟล์ "ก่อน" เขียนแถวใหม่ไว้ก่อน เผื่อต้องลบทีหลัง (BACKSPACE)
            if os.path.exists(output_path):
                last_save_position = os.path.getsize(output_path)
            else:
                last_save_position = 0

            row = landmarks_to_list(dominant_hand)
            # เปิดไฟล์แบบ "a" (append) คือเขียนต่อท้ายไฟล์ ไม่ลบของเดิม
            with open(output_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            saved_count += 1
            print(f"บันทึกแล้ว! ตอนนี้ตัว '{current_label}' มีข้อมูล {saved_count} ครั้ง")

    if key == 8:  # 8 คือรหัสของปุ่ม BACKSPACE
        if last_save_position is None:
            print("ไม่มีท่าล่าสุดให้ลบ (ลบได้แค่ท่าที่เพิ่งกด SPACE ไปในรอบนี้เท่านั้น)")
        else:
            # ตัดไฟล์ให้เหลือแค่ถึงตำแหน่งก่อนหน้าที่เคยบันทึกไว้ = ลบแถวล่าสุดทิ้ง
            with open(output_path, "r+", newline="") as f:
                f.truncate(last_save_position)
            saved_count -= 1
            print(f"ลบท่าล่าสุดแล้ว ตอนนี้ตัว '{current_label}' เหลือข้อมูล {saved_count} ครั้ง")
            last_save_position = None  # ลบได้แค่ครั้งเดียวติดกัน ป้องกันกดรัวจนลบเกิน

# ---------- เก็บกวาดก่อนปิดโปรแกรม ----------
cap.release()
cv2.destroyAllWindows()
print(f"\nจบการอัดข้อมูลตัว '{current_label}' รวมทั้งหมด {saved_count} ครั้ง")
print("รันโปรแกรมนี้ใหม่อีกครั้งเพื่ออัดตัวถัดไป จนกว่าจะครบทุกตัวใน labels.py")
