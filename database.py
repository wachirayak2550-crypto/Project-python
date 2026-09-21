"""
database.py — รวมฟังก์ชันทั้งหมดที่คุยกับฐานข้อมูล SQLite

ใช้ SQLite เพราะมากับ Python อยู่แล้ว (import sqlite3 ได้เลย) ไม่ต้องติดตั้ง/ตั้งเซิร์ฟเวอร์เพิ่ม
ฐานข้อมูลทั้งหมดเก็บอยู่ในไฟล์เดียวชื่อ handspeak.db (จะถูกสร้างอัตโนมัติตอนรันครั้งแรก)

มี 4 ตาราง:
- users            บัญชีผู้ใช้ (สมาชิก + แอดมิน) สำหรับ login/สมัครสมาชิก
- labels           คำ/เลขที่ระบบภาษามือรู้จัก (แทนที่ labels.py แบบเดิม)
- predictions_log  ประวัติการทายผลทุกครั้งที่เกิดขึ้นจริงตอนใช้งาน
- sentences        ประโยคที่ผู้ใช้กดบันทึกไว้

ไฟล์อื่น (app.py, 2_collect_data.py, 3_train_model.py) จะ import ฟังก์ชันจากที่นี่ไปใช้
"""

import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "handspeak.db"


def get_connection():
    """เปิดการเชื่อมต่อฐานข้อมูล 1 ครั้ง (เรียกใช้ทุกครั้งที่จะอ่าน/เขียนข้อมูล)"""
    return sqlite3.connect(DB_PATH)


def hash_password(password):
    """
    เข้ารหัสรหัสผ่านก่อนเก็บลงฐานข้อมูล (ห้ามเก็บรหัสผ่านตรงๆ เป็นตัวหนังสือธรรมดา)
    ใช้ sha256 ซึ่งมากับ Python อยู่แล้ว (ไม่ต้องติดตั้งอะไรเพิ่ม)
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def now_text():
    """คืนค่าเวลาปัจจุบันเป็นตัวหนังสือ เอาไว้บันทึกลงคอลัมน์ created_at / predicted_at"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# ตั้งค่าฐานข้อมูลครั้งแรก (สร้างตาราง + ข้อมูลตั้งต้น)
# ============================================================

def create_tables():
    """สร้างตารางทั้ง 4 ตัว ถ้ายังไม่มี (เรียกซ้ำได้ทุกครั้งที่เปิดโปรแกรม ไม่ลบของเดิม)"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            label_name TEXT NOT NULL,
            predicted_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sentence_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def seed_admin_account(username="admin", password="admin1234"):
    """
    สร้างบัญชีแอดมินเริ่มต้นไว้ล่วงหน้า (ถ้ายังไม่มี)
    ตั้งใจให้สร้างจากตรงนี้เท่านั้น ไม่เปิดให้สมัครเป็น admin ผ่านหน้าเว็บ
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    already_exists = cursor.fetchone() is not None

    if not already_exists:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
            (username, hash_password(password), now_text()),
        )
        conn.commit()

    conn.close()
    return not already_exists  # True ถ้าเพิ่งสร้างใหม่


def migrate_labels_from_file():
    """
    ย้ายรายชื่อคำจาก labels.py (ของเดิม) เข้าตาราง labels
    คำที่มีอยู่แล้วในตารางจะถูกข้าม ไม่ซ้ำ (ใช้ INSERT OR IGNORE)
    """
    from labels import LABELS

    conn = get_connection()
    cursor = conn.cursor()

    for name in LABELS:
        category = "number" if name.isdigit() else "word"
        cursor.execute(
            "INSERT OR IGNORE INTO labels (name, category, created_at) VALUES (?, ?, ?)",
            (name, category, now_text()),
        )

    conn.commit()
    conn.close()


def sync_labels_with_file():
    """
    ทำให้ตาราง labels ตรงกับ labels.py เป๊ะๆ เสมอ (เพิ่มคำที่ขาด + ลบคำที่ตัดออกไปแล้ว)

    เหตุผลที่ต้องมีฟังก์ชันนี้: ทุกคนมีไฟล์ handspeak.db ของตัวเอง (ไม่ sync ผ่าน git)
    ถ้าใครตั้งเครื่องไว้ตั้งแต่ก่อนที่ทีมจะปรับสโคปคำ (เช่น ตัดคำออกทีหลัง)
    ฐานข้อมูลเครื่องนั้นจะยังค้างคำเก่าอยู่ ฟังก์ชันนี้จะลบคำเก่าที่ไม่อยู่ใน labels.py แล้ว
    ออกให้อัตโนมัติทุกครั้งที่รันโปรแกรม ไม่ต้องลบไฟล์ handspeak.db เองอีกต่อไป
    """
    from labels import LABELS

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM labels")
    existing_names = {row[0] for row in cursor.fetchall()}
    wanted_names = set(LABELS)

    # วนตาม "ลิสต์ LABELS" ตามลำดับเป๊ะๆ (ห้ามวนตาม set เพราะ Python เรียงลำดับ
    # ของ set แบบสุ่มไม่คงที่ในแต่ละเครื่อง/แต่ละครั้งที่รัน จะทำให้ id ที่ auto-increment
    # ได้ไม่ตรงกันข้ามเครื่อง แล้ว index [0][1][2]... ที่โชว์ตอนอัดข้อมูลจะไม่ตรงกันไปด้วย)
    for name in LABELS:
        if name not in existing_names:
            category = "number" if name.isdigit() else "word"
            cursor.execute(
                "INSERT INTO labels (name, category, created_at) VALUES (?, ?, ?)",
                (name, category, now_text()),
            )

    for name in existing_names - wanted_names:
        cursor.execute("DELETE FROM labels WHERE name = ?", (name,))

    conn.commit()
    conn.close()


def ensure_ready():
    """
    เตรียมฐานข้อมูลให้พร้อมใช้งานอัตโนมัติทุกครั้งที่รันโปรแกรม
    (เช่น เพิ่ง git clone มาใหม่ยังไม่มี handspeak.db เลย หรือทีมเพิ่งปรับสโคปคำ)

    สร้างตารางถ้ายังไม่มี สร้างบัญชีแอดมินเริ่มต้นถ้ายังไม่มีใครเลย
    แล้วซิงค์ตาราง labels ให้ตรงกับ labels.py เสมอ (เพิ่มคำใหม่ + ลบคำที่ตัดออกไปแล้ว)
    ไฟล์อื่น (2_collect_data.py, 3_train_model.py, app.py) เรียกใช้ฟังก์ชันนี้ที่เดียวพอ
    """
    create_tables()
    if len(get_all_users()) == 0:
        seed_admin_account()
    sync_labels_with_file()


# ============================================================
# users — สมัครสมาชิก / login / ดูรายชื่อ
# ============================================================

def create_user(username, password):
    """สมัครสมาชิกใหม่ (role เป็น 'user' เสมอ) คืนค่า (สำเร็จไหม, ข้อความ)"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
            (username, hash_password(password), now_text()),
        )
        conn.commit()
        return True, "สมัครสมาชิกสำเร็จ ลองเข้าสู่ระบบได้เลย"
    except sqlite3.IntegrityError:
        return False, "มีชื่อผู้ใช้นี้อยู่แล้ว ลองชื่ออื่น"
    finally:
        conn.close()


def verify_user(username, password):
    """
    เช็ค username/password กับฐานข้อมูล
    ถ้าถูกต้อง คืนค่า dict {id, username, role} ถ้าผิด คืนค่า None
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, role FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password)),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return {"id": row[0], "username": username, "role": row[1]}


def get_all_users():
    """ดึงรายชื่อสมาชิกทั้งหมด (ไม่ดึง password มาด้วย เพื่อความปลอดภัย)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, created_at FROM users ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return rows


# ============================================================
# labels — คลังคำที่ระบบรู้จัก (CRUD สำหรับหน้าแอดมิน)
# ============================================================

def get_all_labels():
    """ดึงข้อมูล labels ทั้งหมด (ใช้แสดงในหน้าแอดมิน)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, created_at FROM labels ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_label_names():
    """คืนแค่ list ชื่อคำ (ใช้แทน LABELS จาก labels.py เดิม ในไฟล์อื่นๆ)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM labels ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def add_label(name, category):
    """เพิ่มคำใหม่ คืนค่า (สำเร็จไหม, ข้อความ)"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO labels (name, category, created_at) VALUES (?, ?, ?)",
            (name, category, now_text()),
        )
        conn.commit()
        return True, "เพิ่มคำสำเร็จ"
    except sqlite3.IntegrityError:
        return False, "มีคำนี้อยู่แล้ว"
    finally:
        conn.close()


def update_label(label_id, new_name, new_category):
    """แก้ไขชื่อ/ประเภทของคำที่มี id ตรงกับ label_id"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE labels SET name = ?, category = ? WHERE id = ?",
        (new_name, new_category, label_id),
    )
    conn.commit()
    conn.close()


def delete_label(label_id):
    """ลบคำที่มี id ตรงกับ label_id"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM labels WHERE id = ?", (label_id,))
    conn.commit()
    conn.close()


# ============================================================
# predictions_log — ประวัติการทายผล
# ============================================================

def log_prediction(user_id, label_name):
    """บันทึกว่ามีการทายคำนี้ได้ 1 ครั้ง (user_id เป็น None ได้ถ้าเป็น guest)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions_log (user_id, label_name, predicted_at) VALUES (?, ?, ?)",
        (user_id, label_name, now_text()),
    )
    conn.commit()
    conn.close()


def get_prediction_counts():
    """นับว่าคำไหนถูกทายไปกี่ครั้ง เรียงจากมากไปน้อย (เอาไว้ทำกราฟในหน้าแอดมิน)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT label_name, COUNT(*) as total
        FROM predictions_log
        GROUP BY label_name
        ORDER BY total DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_recent_predictions(limit=50):
    """ดึงประวัติการทายผลล่าสุด (เรียงใหม่สุดก่อน)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT label_name, predicted_at FROM predictions_log ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# ============================================================
# sentences — ประโยคที่ผู้ใช้กดบันทึกไว้
# ============================================================

def save_sentence(user_id, sentence_text):
    """บันทึกประโยคที่ผู้ใช้กดบันทึกไว้ตอนใช้งาน"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sentences (user_id, sentence_text, created_at) VALUES (?, ?, ?)",
        (user_id, sentence_text, now_text()),
    )
    conn.commit()
    conn.close()


def get_all_sentences():
    """ดึงประโยคที่เคยบันทึกไว้ทั้งหมด (ใหม่สุดก่อน)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT sentence_text, created_at FROM sentences ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
