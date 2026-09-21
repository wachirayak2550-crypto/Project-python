"""
STEP 4: ทายสดจากกล้อง + แสดงผลเป็นข้อความตัวใหญ่บนจอ

โปรแกรมนี้จะ:
1. เปิดกล้อง จับมือด้วย MediaPipe (เหมือน STEP 1-2)
2. เอาจุดมือ 63 ตัวเลขของมือที่เด่นสุด ไปให้โมเดล (model.pkl) ทายว่าเป็นตัวอะไร
3. โชว์คำตอบเป็นตัวหนังสือขนาดใหญ่บนจอ
4. กันไม่ให้ข้อความกระพริบถี่เกินไป โดยจะเปลี่ยนข้อความก็ต่อเมื่อ
   โมเดลทายผลเดิมซ้ำกันต่อเนื่องหลายเฟรม (แปลว่ามือ "นิ่ง" จริงๆ ไม่ใช่แค่ผ่านมือไวๆ)
5. เก็บคำที่ทายได้แต่ละคำ "ต่อกันเป็นประโยค" แสดงไว้แถวล่างของจอด้วย
   (พอมือนิ่งจนคำเปลี่ยนแล้ว จะเติมคำนั้นเข้าประโยคอัตโนมัติ 1 ครั้งต่อ 1 ท่า)

ก่อนรันไฟล์นี้ ต้องมี model.pkl แล้ว (ได้มาจากการรัน 3_train_model.py)

วิธีใช้: รัน python 4_predict_live.py แล้วทำท่ามือหน้ากล้อง
กด 'c' เพื่อล้างประโยคทั้งหมด, กด BACKSPACE เพื่อลบคำล่าสุดออกจากประโยค
กด 'q' เพื่อออกจากโปรแกรม
"""

import os
import pickle

import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

MODEL_PATH = "model.pkl"

# ต้องทายผลเดิมซ้ำกันกี่เฟรมติดกัน ถึงจะยอมเปลี่ยนข้อความที่แสดงบนจอ
# ค่ายิ่งสูง = ข้อความนิ่งขึ้น แต่จะตอบสนองช้าลงนิดหน่อย
STABLE_FRAMES_REQUIRED = 8

# cv2.putText ธรรมดา "วาดภาษาไทยไม่ได้" (จะขึ้นเป็น ??? แทน) เพราะ label ตอนนี้มีทั้งเลขและคำไทย
# เลยต้องใช้ Pillow (PIL) ช่วยวาดตัวหนังสือไทยแทน โดยใช้ฟอนต์ Tahoma ที่มากับ Windows
THAI_FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"
thai_font_large = ImageFont.truetype(THAI_FONT_PATH, 80)
thai_font_small = ImageFont.truetype(THAI_FONT_PATH, 28)


def draw_thai_text(frame, text, position, font=thai_font_small, color_bgr=(0, 255, 0)):
    """
    วาดข้อความภาษาไทย (หรือข้อความอะไรก็ได้) ลงบนภาพ frame
    วิธีทำ: แปลงภาพเป็นรูปแบบของ Pillow -> วาดตัวหนังสือ -> แปลงกลับเป็นภาพของ OpenCV
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # PIL ใช้สีแบบ RGB ไม่ใช่ BGR
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- เช็คว่ามีโมเดลหรือยัง ----------
if not os.path.exists(MODEL_PATH):
    print(f"ไม่พบไฟล์ {MODEL_PATH} กรุณารัน 3_train_model.py ก่อน เพื่อเทรนและเซฟโมเดล")
    exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- เตรียมตัวจับมือของ MediaPipe (เหมือน STEP 1-2) ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def hand_bbox_area(hand_landmarks):
    """เหมือนใน 2_collect_data.py: ใช้หาว่ามือไหนใหญ่/เด่นสุดในเฟรม"""
    xs = [point.x for point in hand_landmarks.landmark]
    ys = [point.y for point in hand_landmarks.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def landmarks_to_list(hand_landmarks):
    """เหมือนใน 2_collect_data.py: แปลงจุดมือ 21 จุด เป็นตัวเลข 63 ค่า"""
    row = []
    for point in hand_landmarks.landmark:
        row.append(point.x)
        row.append(point.y)
        row.append(point.z)
    return row


# ---------- เปิดกล้อง ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("เปิดกล้องไม่ได้ ลองเช็คว่ามีโปรแกรมอื่นใช้กล้องอยู่หรือเปล่า")
    exit()

print("พร้อมทายแล้ว! ทำท่ามือหน้ากล้องได้เลย")
print("กด 'q' เพื่อออกจากโปรแกรม")

displayed_label = "-"     # ข้อความที่กำลังแสดงอยู่บนจอตอนนี้
candidate_label = None    # ผลทายล่าสุดที่กำลังรอดูว่านิ่งพอจะขึ้นจอไหม
candidate_count = 0       # นับว่า candidate_label ซ้ำกันมาแล้วกี่เฟรมติดกัน

sentence_words = []       # ลิสต์เก็บคำที่ยืนยันแล้ว เรียงกันเป็นประโยค
already_added = False     # กันไม่ให้เติมคำเดิมซ้ำๆ ขณะที่ยังค้างท่าเดิมอยู่

while True:
    ret, frame = cap.read()
    if not ret:
        print("อ่านภาพจากกล้องไม่ได้ หยุดโปรแกรม")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    dominant_hand = None
    best_area = -1

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            area = hand_bbox_area(hand_landmarks)
            if area > best_area:
                best_area = area
                dominant_hand = hand_landmarks

    # ---------- ทายผลของเฟรมนี้ ----------
    current_prediction = None
    if dominant_hand is not None:
        # model.predict ต้องการข้อมูลเป็น "list ของแถว" เลยต้องห่อด้วย [ ] อีกชั้น (2 มิติ)
        features = np.array([landmarks_to_list(dominant_hand)])
        current_prediction = model.predict(features)[0]

    # ---------- เช็คว่าผลทายนิ่งพอจะเปลี่ยนข้อความบนจอไหม ----------
    if current_prediction == candidate_label:
        candidate_count += 1
    else:
        candidate_label = current_prediction
        candidate_count = 1
        already_added = False  # ท่าเปลี่ยนแล้ว เปิดสิทธิ์ให้เติมคำใหม่เข้าประโยคได้อีกครั้ง

    if candidate_count >= STABLE_FRAMES_REQUIRED:
        displayed_label = candidate_label if candidate_label is not None else "-"

        # ท่านิ่งพอแล้ว และยังไม่เคยเติมคำนี้เข้าประโยคระหว่างค้างท่านี้ -> เติมเข้าประโยค
        if candidate_label is not None and not already_added:
            sentence_words.append(candidate_label)
            already_added = True

    # ---------- วาดข้อความตัวใหญ่บนจอ (คำที่ทายได้ตอนนี้) ----------
    frame = draw_thai_text(frame, str(displayed_label), (30, 230), font=thai_font_large)

    # ---------- วาดแถวประโยคที่สะสมไว้ ----------
    sentence_text = " ".join(sentence_words)
    frame = draw_thai_text(frame, sentence_text, (10, 420), color_bgr=(255, 255, 0))

    if dominant_hand is None:
        frame = draw_thai_text(frame, "ไม่เจอมือ - ยกมือเข้ามาในเฟรม", (10, 10), color_bgr=(0, 0, 255))

    frame = draw_thai_text(frame, "c = clear, BACKSPACE = undo word, q = quit", (10, 45))
    cv2.imshow("HandSpeak - Predict Live", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        # ล้างประโยคทั้งหมด เริ่มนับใหม่
        sentence_words = []

    if key == 8:  # BACKSPACE
        # ลบคำล่าสุดออกจากประโยค (ถ้ามี)
        if sentence_words:
            sentence_words.pop()

# ---------- เก็บกวาดก่อนปิดโปรแกรม ----------
cap.release()
cv2.destroyAllWindows()
