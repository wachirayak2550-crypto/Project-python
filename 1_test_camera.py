"""
STEP 1: ทดสอบกล้อง + จับจุดมือด้วย MediaPipe

โปรแกรมนี้แค่เปิดกล้อง แล้ววาดจุด 21 จุดบนมือที่จับได้ขึ้นจอ
รองรับการจับพร้อมกันได้สูงสุด 2 มือ (ยกมือเดียวหรือ 2 มือก็ได้)
ยังไม่มีการทายเลขอะไรทั้งสิ้น แค่เช็คว่ากล้องกับ MediaPipe ทำงานได้จริง

หมายเหตุ: ตอนอัดข้อมูล/ทายจริง (STEP 2 เป็นต้นไป) เราจะใช้แค่ "มือเดียว"
(มือที่เด่นสุดในเฟรม) ป้อนให้โมเดล เพื่อให้จำนวนตัวเลขคงที่เสมอ
แต่ตอนนี้ในหน้าจอทดสอบ เราวาดจุดให้เห็นได้ทั้ง 2 มือ เพื่อให้ demo ดูสมบูรณ์

วิธีใช้: รัน python 1_test_camera.py แล้วยกมือหน้ากล้อง
กด 'q' บนคีย์บอร์ดเพื่อปิดโปรแกรม
"""

import cv2
import mediapipe as mp

# ---------- เตรียมตัวจับมือของ MediaPipe ----------
# mp_hands คือโมดูลที่มีความสามารถจับจุดบนมือ (hand landmarks)
mp_hands = mp.solutions.hands

# mp_drawing ใช้สำหรับวาดจุดและเส้นเชื่อมบนมือให้เราเห็นบนจอ
mp_drawing = mp.solutions.drawing_utils

# สร้างตัวจับมือ (Hands object)
# - static_image_mode=False  -> บอกว่าเราใช้กับวิดีโอสด (ไม่ใช่ภาพนิ่ง) จะได้ตามมือได้ลื่นขึ้น
# - max_num_hands=2          -> จับได้สูงสุด 2 มือ (เผื่อผู้ใช้ยกมือซ้ายหรือขวา และดู demo สมบูรณ์)
# - min_detection_confidence -> ความมั่นใจขั้นต่ำว่าเจอมือจริง (0.0 - 1.0)
# - min_tracking_confidence  -> ความมั่นใจขั้นต่ำตอนตามมือที่เจอแล้วในเฟรมถัดไป
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ---------- เปิดกล้อง ----------
# เลข 0 หมายถึงกล้องตัวแรกของเครื่อง (ปกติคือกล้องในตัวโน้ตบุ๊ค)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("เปิดกล้องไม่ได้ ลองเช็คว่ามีโปรแกรมอื่นใช้กล้องอยู่หรือเปล่า")
    exit()

print("เปิดกล้องสำเร็จ! ยกมือขึ้นมาหน้ากล้องได้เลย")
print("กด 'q' ที่คีย์บอร์ดเพื่อปิดโปรแกรม")

# ---------- ลูปอ่านภาพจากกล้องทีละเฟรม ----------
while True:
    # อ่านภาพ 1 เฟรมจากกล้อง
    # ret = True ถ้าอ่านสำเร็จ, frame = ภาพที่ได้ (เป็นตาราง pixel สี BGR)
    ret, frame = cap.read()

    if not ret:
        print("อ่านภาพจากกล้องไม่ได้ หยุดโปรแกรม")
        break

    # กลับภาพซ้าย-ขวา (mirror) ให้เหมือนส่องกระจก จะได้ใช้งานง่ายขึ้น
    frame = cv2.flip(frame, 1)

    # MediaPipe ต้องการภาพสีแบบ RGB แต่ OpenCV อ่านภาพมาเป็น BGR
    # เลยต้องแปลงสีก่อนส่งให้ MediaPipe ประมวลผล
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ส่งภาพให้ MediaPipe ตรวจจับมือ -> ได้ผลลัพธ์กลับมาเก็บใน result
    result = hands.process(rgb_frame)

    # ถ้าเจอมือในภาพ (result.multi_hand_landmarks ไม่ว่าง)
    if result.multi_hand_landmarks:
        # วนดูทีละมือที่เจอ (อาจเจอ 1 หรือ 2 มือ เพราะตั้ง max_num_hands=2)
        for hand_landmarks in result.multi_hand_landmarks:
            # วาดจุด 21 จุด + เส้นเชื่อมระหว่างจุด ลงบนภาพ frame โดยตรง
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

    # แสดงภาพผลลัพธ์ในหน้าต่างชื่อ "HandSpeak - Test Camera"
    cv2.imshow("HandSpeak - Test Camera", frame)

    # รอรับปุ่มกดคีย์บอร์ด 1 มิลลิวินาที
    # ถ้ากด 'q' ให้ออกจากลูป (ปิดโปรแกรม)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- เก็บกวาดก่อนปิดโปรแกรม ----------
cap.release()          # คืนกล้องให้ระบบ
cv2.destroyAllWindows()  # ปิดหน้าต่างที่เปิดไว้ทั้งหมด
