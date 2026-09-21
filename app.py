"""
STEP 7-9: หน้าเว็บ HandSpeak แบบเต็มรูปแบบ (login/สมัครสมาชิก + ใช้งานจริง + แอดมิน)

โครงสร้างของไฟล์นี้: มี 3 "หน้าจอ" สลับกันแสดงตาม st.session_state.user
(ไม่ได้ใช้ระบบ multi-page ของ Streamlit เพราะต้องสลับหน้าตามสิทธิ์ผู้ใช้ ทำแบบ if/else ธรรมดาเข้าใจง่ายกว่า)

1. show_login_page()  -> ยังไม่ login (st.session_state.user is None)
   มีฟอร์ม login, ฟอร์มสมัครสมาชิก, และปุ่มเข้าใช้งานแบบไม่ login (Guest)

2. show_admin_page()  -> login แล้วและ role เป็น "admin"
   จัดการคำศัพท์ (CRUD), ดูประวัติการใช้งาน, ดูรายชื่อสมาชิก

3. show_main_page()   -> login แล้ว (role "user") หรือเข้าแบบ Guest
   หน้าใช้งานจริง: เปิดกล้อง -> จับมือ -> ทายผล -> สะสมเป็นประโยค (เหมือน STEP 4-5 เดิม)
   ทุกครั้งที่ทายคำได้ หรือกดบันทึกประโยค จะเขียนลงฐานข้อมูลจริง

ก่อนรันไฟล์นี้ ต้องรัน 0_setup_database.py ก่อน (สร้างฐานข้อมูล + บัญชีแอดมินเริ่มต้น)
และต้องมี model.pkl แล้ว (ได้มาจากการรัน 3_train_model.py) ถึงจะใช้หน้าใช้งานจริงได้

วิธีใช้: รัน streamlit run app.py
"""

import os
import pickle
import threading
import time

import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import database as db

# เช็คก่อนว่า mediapipe เวอร์ชันนี้มีฟีเจอร์จับมือที่เราต้องใช้ไหม
# (ถ้าเครื่องใช้ Python ใหม่เกินไป เช่น 3.12/3.13 อาจได้ mediapipe เวอร์ชันที่ตัดฟีเจอร์นี้ทิ้งไปแล้ว)
if not hasattr(mp, "solutions"):
    st.error(
        "เจอปัญหา: mediapipe เวอร์ชันนี้ไม่มีฟีเจอร์จับมือที่โปรแกรมนี้ต้องใช้\n\n"
        "สาเหตุที่พบบ่อยที่สุด: เครื่องนี้ใช้ Python เวอร์ชันใหม่เกินไป (เช่น 3.12 หรือ 3.13)\n\n"
        "**วิธีแก้:**\n"
        "1. ติดตั้ง Python 3.11 จาก https://www.python.org/downloads/release/python-3119/\n"
        "2. ติดตั้ง library ใหม่ด้วย Python 3.11: `py -3.11 -m pip install -r requirements.txt`\n"
        "3. รันเว็บใหม่ด้วย Python 3.11: `py -3.11 -m streamlit run app.py`"
    )
    st.stop()

MODEL_PATH = "model.pkl"

# ต้องทายผลเดิมซ้ำกันกี่เฟรมติดกัน ถึงจะยอมเปลี่ยนข้อความที่แสดง (กันกระพริบ เหมือน STEP 4)
STABLE_FRAMES_REQUIRED = 8

# เตรียมฐานข้อมูลให้พร้อม (สร้างตาราง + ย้ายคำจาก labels.py อัตโนมัติถ้าเป็นเครื่องใหม่)
# กันเผื่อลืมรัน 0_setup_database.py มาก่อน (เช่น เพิ่ง git clone มา)
db.ensure_ready()


# ============================================================
# ธีมหน้าตา — ก็อปสี/ฟอนต์มาจากไฟล์ที่เพื่อนออกแบบ (n1.html, n2.html)
# ใช้กับหน้า Login และหน้าใช้งานจริงเท่านั้น (หน้าแอดมินไม่ต้องสวย ตามที่ตกลงกันไว้)
# ============================================================

PINK = "#eb0f67"
ORANGE = "#e16347"
TEXT_DARK = "#243b6b"
TEXT_MUTED = "#7183a7"
BORDER = "#dce4f3"


def apply_theme():
    """แปะ CSS ทับหน้าตาเริ่มต้นของ Streamlit ให้เข้าธีมสีที่เพื่อนออกแบบไว้"""
    st.markdown(f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at 10% 10%, #e4efff, transparent 25%),
            radial-gradient(circle at 90% 90%, #e9e2ff, transparent 30%),
            #f7f9ff;
        font-family: "Noto Sans Thai", "Segoe UI", Arial, sans-serif;
    }}
    h1, h2, h3, p, label, span {{ color: {TEXT_DARK}; }}

    .stButton > button {{
        border-radius: 12px;
        border: 1.5px solid {BORDER};
        background: white;
        color: {TEXT_DARK};
        font-weight: 500;
        transition: 0.2s;
    }}
    .stButton > button:hover {{
        border-color: {PINK};
        color: {PINK};
    }}
    .stButton > button[kind="primary"] {{
        background: {PINK};
        border: none;
        color: white;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.9;
        color: white;
    }}

    .stTextInput input {{
        border-radius: 12px;
        border: 1.5px solid {BORDER};
        background: white !important;
        color: {TEXT_DARK} !important;
    }}
    .stTextInput input:focus {{
        border-color: #c0caec;
        box-shadow: 0 0 0 4px rgba(99, 130, 238, 0.1);
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {{
        color: {PINK} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {PINK} !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 20px !important;
        box-shadow: 0 20px 50px rgba(45, 75, 130, 0.15);
        background: white !important;
    }}

    [data-testid="stImage"] img {{
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_logo(size_px=50, centered=True):
    """โลโก้ Hand(สีชมพู)Speak(สีส้ม) ตามดีไซน์ของเพื่อน"""
    align = "center" if centered else "left"
    st.markdown(f"""
    <div style="text-align:{align}; margin-bottom:10px;">
        <h1 style="font-size:{size_px}px; margin:0; color:{PINK};">
            Hand<span style="color:{ORANGE};">Speak</span>
        </h1>
        <p style="color:{TEXT_MUTED}; font-size:16px; margin-top:5px;">ระบบล่ามภาษามือ</p>
    </div>
    """, unsafe_allow_html=True)


def hand_bbox_area(hand_landmarks):
    """หาว่ามือไหนใหญ่/เด่นสุดในเฟรม (เหมือนไฟล์ STEP 2 และ STEP 4)"""
    xs = [point.x for point in hand_landmarks.landmark]
    ys = [point.y for point in hand_landmarks.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def landmarks_to_list(hand_landmarks):
    """แปลงจุดมือ 21 จุด เป็นตัวเลข 63 ค่า (เหมือนไฟล์ STEP 2 และ STEP 4)"""
    row = []
    for point in hand_landmarks.landmark:
        row.append(point.x)
        row.append(point.y)
        row.append(point.z)
    return row


# ต้องมี STUN server ตัวนี้ให้กล้องของ "คนที่เปิดเว็บ" เชื่อมต่อกับเซิร์ฟเวอร์ของเราได้
# (จำเป็นเวลา deploy ขึ้นเว็บจริง ไม่งั้นเบราว์เซอร์กับเซิร์ฟเวอร์จะหากันไม่เจอ)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


class HandSignProcessor(VideoProcessorBase):
    """
    ตัวประมวลผลวิดีโอของ streamlit-webrtc — รับภาพจาก "กล้องของคนที่เปิดเว็บ"
    (ไม่ใช่กล้องเซิร์ฟเวอร์แบบ cv2.VideoCapture เดิม) มาประมวลผลทีละเฟรม

    ทำงานเหมือน show_main_page เวอร์ชันเดิมทุกอย่าง (จับมือ -> ทายผล -> เช็คว่านิ่งพอไหม)
    แค่โค้ดส่วนนี้รันอยู่คนละ thread กับหน้าเว็บหลัก เลยต้องมี self.lock
    กันไม่ให้ 2 thread อ่าน/เขียนตัวแปรเดียวกันพร้อมกันจนข้อมูลปนกัน
    """

    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.lock = threading.Lock()
        self.displayed_label = "-"
        self.candidate_label = None
        self.candidate_count = 0
        self.already_added = False
        # คำใหม่ที่เพิ่งนิ่งพอ รอให้หน้าเว็บหลักมาหยิบไปเติมประโยค (เคลียร์เป็น None หลังหยิบไปแล้ว)
        self.pending_new_word = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_img)

        dominant_hand = None
        best_area = -1
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                area = hand_bbox_area(hand_landmarks)
                if area > best_area:
                    best_area = area
                    dominant_hand = hand_landmarks

        current_prediction = None
        if dominant_hand is not None:
            features = np.array([landmarks_to_list(dominant_hand)])
            current_prediction = self.model.predict(features)[0]

        with self.lock:
            if current_prediction == self.candidate_label:
                self.candidate_count += 1
            else:
                self.candidate_label = current_prediction
                self.candidate_count = 1
                self.already_added = False

            if self.candidate_count >= STABLE_FRAMES_REQUIRED:
                self.displayed_label = self.candidate_label if self.candidate_label is not None else "-"
                if self.candidate_label is not None and not self.already_added:
                    self.already_added = True
                    self.pending_new_word = self.candidate_label

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# หน้าจอที่ 1: Login / สมัครสมาชิก
# ============================================================

def show_login_page():
    apply_theme()

    # จำกัดความกว้างให้เหมือนการ์ดกลางจอตามดีไซน์ของเพื่อน (n1.html)
    _, center_col, _ = st.columns([1, 2, 1])

    with center_col:
        render_logo(size_px=50, centered=True)

        with st.container(border=True):
            tab_login, tab_signup = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

            with tab_login:
                st.markdown(f"<h3 style='margin-bottom:0;'>Welcome back!</h3>", unsafe_allow_html=True)
                st.caption("เข้าสู่ระบบเพื่อใช้งาน HandSpeak")

                username = st.text_input("Username", key="login_username", placeholder="Username")
                password = st.text_input("Password", type="password", key="login_password", placeholder="Password")
                if st.button("Log in", type="primary", use_container_width=True):
                    user = db.verify_user(username, password)
                    if user is not None:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Username หรือ Password ไม่ถูกต้อง")

            with tab_signup:
                st.markdown(f"<h3 style='margin-bottom:0;'>Create Account</h3>", unsafe_allow_html=True)
                st.caption("สมัครสมาชิกเพื่อใช้งาน HandSpeak")

                new_username = st.text_input("Username ใหม่", key="signup_username", placeholder="Username")
                new_password = st.text_input("Password ใหม่", type="password", key="signup_password", placeholder="Password")
                confirm_password = st.text_input("ยืนยัน Password", type="password", key="signup_confirm", placeholder="Confirm Password")
                if st.button("Sign in", type="primary", use_container_width=True):
                    if not new_username or not new_password:
                        st.error("กรุณากรอก Username และ Password ให้ครบ")
                    elif new_password != confirm_password:
                        st.error("Password ทั้งสองช่องไม่ตรงกัน")
                    else:
                        ok, msg = db.create_user(new_username, new_password)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

        st.write("")
        if st.button("👥  เข้าใช้งานแบบไม่ต้อง Login", use_container_width=True):
            # Guest ไม่มี id จริงในฐานข้อมูล ใช้ None แทน (predictions_log/sentences จะรับ user_id ว่างได้)
            st.session_state.user = {"id": None, "username": "Guest", "role": "user"}
            st.rerun()


# ============================================================
# หน้าจอที่ 2: แอดมิน (ไม่เน้นความสวยงาม เน้นใช้งานได้ครบ)
# ============================================================

def show_admin_page():
    user = st.session_state.user

    top_col1, top_col2 = st.columns([4, 1])
    top_col1.title("🛠️ HandSpeak - หน้าแอดมิน")
    if top_col2.button("ออกจากระบบ"):
        st.session_state.user = None
        st.rerun()
    st.write(f"เข้าสู่ระบบในชื่อ: {user['username']}")

    tab_labels, tab_history, tab_users = st.tabs(
        ["จัดการคำศัพท์", "ประวัติการใช้งาน", "รายชื่อสมาชิก"]
    )

    # ---------- แท็บ: จัดการคำศัพท์ (CRUD) ----------
    with tab_labels:
        st.subheader("คำศัพท์ทั้งหมด")
        labels = db.get_all_labels()
        label_rows = [
            {"ID": l[0], "คำ": l[1], "ประเภท": l[2], "วันที่เพิ่ม": l[3]} for l in labels
        ]
        st.dataframe(label_rows, width='stretch')

        st.subheader("เพิ่มคำใหม่")
        col1, col2 = st.columns(2)
        new_name = col1.text_input("ชื่อคำ", key="new_label_name")
        new_category = col2.selectbox("ประเภท", ["number", "word"], key="new_label_category")
        if st.button("เพิ่มคำ"):
            if new_name:
                ok, msg = db.add_label(new_name, new_category)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("กรุณากรอกชื่อคำ")

        st.subheader("แก้ไข / ลบคำ")
        if labels:
            label_options = {f"{l[1]} (id={l[0]})": l for l in labels}
            selected_key = st.selectbox("เลือกคำ", list(label_options.keys()))
            selected_label = label_options[selected_key]

            edit_name = st.text_input("ชื่อใหม่", value=selected_label[1], key="edit_label_name")
            edit_category = st.selectbox(
                "ประเภทใหม่", ["number", "word"],
                index=0 if selected_label[2] == "number" else 1,
                key="edit_label_category",
            )

            col_a, col_b = st.columns(2)
            if col_a.button("บันทึกการแก้ไข"):
                db.update_label(selected_label[0], edit_name, edit_category)
                st.success("แก้ไขเรียบร้อย")
                st.rerun()
            if col_b.button("ลบคำนี้"):
                db.delete_label(selected_label[0])
                st.success("ลบเรียบร้อย")
                st.rerun()
        else:
            st.info("ยังไม่มีคำในระบบเลย")

    # ---------- แท็บ: ประวัติการใช้งาน ----------
    with tab_history:
        st.subheader("คำที่ถูกทายบ่อยที่สุด")
        counts = db.get_prediction_counts()
        if counts:
            chart_data = {"คำ": [c[0] for c in counts], "จำนวนครั้ง": [c[1] for c in counts]}
            st.bar_chart(chart_data, x="คำ", y="จำนวนครั้ง")
        else:
            st.info("ยังไม่มีข้อมูลการทายผล")

        st.subheader("ประวัติล่าสุด")
        recent = db.get_recent_predictions(50)
        recent_rows = [{"คำ": r[0], "เวลา": r[1]} for r in recent]
        st.dataframe(recent_rows, width='stretch')

        st.subheader("ประโยคที่เคยบันทึกไว้")
        sentences = db.get_all_sentences()
        sentence_rows = [{"ประโยค": s[0], "เวลา": s[1]} for s in sentences]
        st.dataframe(sentence_rows, width='stretch')

    # ---------- แท็บ: รายชื่อสมาชิก ----------
    with tab_users:
        st.subheader("สมาชิกทั้งหมด")
        users = db.get_all_users()
        user_rows = [
            {"ID": u[0], "Username": u[1], "สิทธิ์": u[2], "วันที่สมัคร": u[3]} for u in users
        ]
        st.dataframe(user_rows, width='stretch')


# ============================================================
# หน้าจอที่ 3: ใช้งานจริง (เหมือนเดิมจาก STEP 5 + บันทึกลงฐานข้อมูล)
# ============================================================

def show_main_page():
    apply_theme()
    user = st.session_state.user

    # ---------- หัวข้อ: โลโก้ซ้าย + ชื่อผู้ใช้/ออกจากระบบ ขวา (ตามดีไซน์ n2.html) ----------
    header_left, header_right = st.columns([3, 1])
    with header_left:
        render_logo(size_px=36, centered=False)
    with header_right:
        st.write("")
        st.markdown(
            f"<p style='text-align:right;'>ผู้ใช้: <strong>{user['username']}</strong></p>",
            unsafe_allow_html=True,
        )
        if st.button("ออกจากระบบ", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    st.caption("ทำท่ามือหน้ากล้องค้างไว้นิ่งๆ เพื่อให้ระบบทาย คำที่ทายได้แต่ละคำจะถูกเก็บต่อกันเป็นประโยคด้านล่าง")
    st.divider()

    if not os.path.exists(MODEL_PATH):
        st.error(f"ไม่พบไฟล์ {MODEL_PATH} กรุณารัน 3_train_model.py ก่อน เพื่อเทรนและเซฟโมเดล")
        return

    if "sentence_words" not in st.session_state:
        st.session_state.sentence_words = []

    # ---------- คำที่ทายได้ ----------
    st.markdown(f"<p style='text-align:center; color:{TEXT_MUTED}; margin-top:20px;'>คำที่ทายได้</p>", unsafe_allow_html=True)
    result_box = st.empty()

    # ---------- ประโยคที่สะสมไว้ ----------
    st.markdown(f"<p style='font-weight:600; margin-top:20px;'>ประโยคที่สะสมไว้</p>", unsafe_allow_html=True)
    sentence_box = st.empty()

    def render_sentence():
        sentence_text = " ".join(st.session_state.sentence_words) if st.session_state.sentence_words else "ยังไม่มีคำ"
        sentence_box.markdown(f"""
            <div style="background:white; border:1px solid {BORDER}; border-radius:10px;
                        padding:18px; color:#52688f; font-size:18px; min-height:30px;">
                {sentence_text}
            </div>
        """, unsafe_allow_html=True)

    render_sentence()

    # ---------- ปุ่มล้าง / ลบ / บันทึก ----------
    col1, col2, col3 = st.columns(3)
    if col1.button("ล้างประโยคทั้งหมด", use_container_width=True):
        st.session_state.sentence_words = []
        render_sentence()
    if col2.button("ลบคำล่าสุด", use_container_width=True):
        if st.session_state.sentence_words:
            st.session_state.sentence_words.pop()
            render_sentence()
    if col3.button("บันทึกประโยคนี้", type="primary", use_container_width=True):
        if st.session_state.sentence_words:
            sentence_text = " ".join(st.session_state.sentence_words)
            db.save_sentence(user["id"], sentence_text)
            st.success(f"บันทึกประโยคแล้ว: {sentence_text}")
        else:
            st.warning("ยังไม่มีคำในประโยคเลย")

    # ---------- กล้อง ----------
    # ใช้ streamlit-webrtc แทน cv2.VideoCapture(0) เดิม เพราะเดิมเปิด "กล้องเซิร์ฟเวอร์"
    # ซึ่งใช้ได้แค่ตอนรันในเครื่องตัวเอง แต่ถ้า deploy ขึ้นเว็บจริง เซิร์ฟเวอร์ไม่มีกล้อง
    # streamlit-webrtc จะขอเปิด "กล้องของคนที่กดลิงก์" ผ่านเบราว์เซอร์แทน ใช้ได้ทั้ง 2 กรณี
    st.markdown("<p style='font-weight:600; margin-top:20px;'>กล้อง</p>", unsafe_allow_html=True)
    st.caption("กดปุ่ม START ด้านล่างเพื่อขอเปิดกล้อง (เบราว์เซอร์จะถามอนุญาตก่อนใช้งานครั้งแรก)")

    ctx = webrtc_streamer(
        key="handspeak-camera",
        video_processor_factory=HandSignProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.state.playing:
        # ลูปนี้จะวนอ่านค่าจาก HandSignProcessor (ที่กำลังประมวลผลอยู่อีก thread หนึ่ง)
        # มาอัปเดตหน้าจอ ทุกๆ 0.3 วินาที จนกว่าจะกด STOP
        while ctx.state.playing:
            if ctx.video_processor:
                with ctx.video_processor.lock:
                    label = ctx.video_processor.displayed_label
                    new_word = ctx.video_processor.pending_new_word
                    ctx.video_processor.pending_new_word = None

                if new_word is not None:
                    st.session_state.sentence_words.append(new_word)
                    db.log_prediction(user["id"], new_word)  # บันทึกประวัติการทายผลลง DB
                    render_sentence()

                result_box.markdown(
                    f"<p style='text-align:center; font-size:40px; color:{PINK}; font-weight:600;'>{label}</p>",
                    unsafe_allow_html=True,
                )
            time.sleep(0.3)
    else:
        result_box.markdown(
            f"<p style='text-align:center; font-size:40px; color:{PINK}; font-weight:600;'>กำลังรอการตรวจจับ...</p>",
            unsafe_allow_html=True,
        )
        st.info("กดปุ่ม START ด้านบนเพื่อเริ่มใช้งาน")


# ============================================================
# จุดเริ่มต้นโปรแกรม: ตัดสินใจว่าจะโชว์หน้าจอไหน
# ============================================================

st.set_page_config(page_title="HandSpeak", page_icon="🤟")

if "user" not in st.session_state:
    st.session_state.user = None  # ยังไม่ login

if st.session_state.user is None:
    show_login_page()
elif st.session_state.user["role"] == "admin":
    show_admin_page()
else:
    show_main_page()
