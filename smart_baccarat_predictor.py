
import streamlit as st
import numpy as np
import joblib

# โหลดโมเดลและ encoder
model = joblib.load("baccarat_smart_model.pkl")
encoder = joblib.load("pattern_encoder.pkl")

# Mapping
option_map = {"Player (P)": 0, "Banker (B)": 1, "Tie (T)": 2}
reverse_map = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}

# ตรวจจับ pattern type
def detect_pattern(p):
    if p == p[::-1]:
        return 'mirror'
    if all(i == p[0] for i in p):
        return 'dragon'
    if len(set(p)) == 2 and p[::2] == p[::2][::-1]:
        return 'pingpong'
    return 'mixed'

# UI
st.title("Smart Baccarat Predictor")
st.write("ทำนายผลโดยวิเคราะห์ 5 ตาหลังสุด พร้อมรูปแบบเกม")

cols = st.columns(5)
inputs = [cols[i].selectbox(f"ผลตาก่อนหน้า {i+1}", list(option_map.keys())) for i in range(5)]
streak = st.number_input("จำนวน streak ล่าสุด", min_value=1, max_value=20, value=1)

if st.button("ทำนายผล"):
    values = [option_map[i] for i in inputs]
    pattern = detect_pattern(values)
    pattern_code = encoder.transform([pattern])[0]
    features = values + [streak, pattern_code]
    prediction = model.predict([features])[0]
    st.success(f"ระบบคาดการณ์ว่า: **{reverse_map[prediction]}**")
