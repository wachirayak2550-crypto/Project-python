"""
STEP 6: ตั้งค่าฐานข้อมูลครั้งแรก

โปรแกรมนี้จะ:
1. สร้างไฟล์ฐานข้อมูล handspeak.db พร้อมตารางทั้ง 4 ตัว (users, labels, predictions_log, sentences)
2. สร้างบัญชีแอดมินเริ่มต้นไว้ล่วงหน้า 1 บัญชี
3. ย้ายรายชื่อคำจาก labels.py (ของเดิม) เข้าตาราง labels

รันไฟล์นี้แค่ครั้งเดียวตอนตั้งค่าโปรเจค (รันซ้ำได้ ไม่ทำให้ข้อมูลเดิมหาย เพราะเช็คก่อนเพิ่มทุกครั้ง)

วิธีใช้: python 0_setup_database.py
"""

import database as db

print("กำลังสร้างตารางฐานข้อมูล...")
db.create_tables()
print("สร้างตารางเรียบร้อย: users, labels, predictions_log, sentences\n")

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin1234"

created = db.seed_admin_account(ADMIN_USERNAME, ADMIN_PASSWORD)
if created:
    print(f"สร้างบัญชีแอดมินแล้ว -> username: {ADMIN_USERNAME}  password: {ADMIN_PASSWORD}")
    print("(แนะนำให้เปลี่ยนรหัสผ่านทีหลังถ้าจะใช้งานจริง)\n")
else:
    print("มีบัญชีแอดมินอยู่แล้ว ข้ามขั้นตอนนี้\n")

print("กำลังย้ายรายชื่อคำจาก labels.py เข้าฐานข้อมูล...")
db.migrate_labels_from_file()

all_labels = db.get_all_labels()
print(f"ตอนนี้ตาราง labels มีข้อมูลทั้งหมด {len(all_labels)} คำ\n")

print("ตั้งค่าฐานข้อมูลเสร็จสมบูรณ์! พร้อมใช้งาน app.py ต่อได้เลย")
