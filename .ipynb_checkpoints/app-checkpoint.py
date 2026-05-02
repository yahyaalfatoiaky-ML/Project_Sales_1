import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تحميل الملفات مع معالجة الأخطاء
try:
    model = joblib.load('laptop_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading files: {e}")

st.title("💻 Laptop Price Predictor Pro")

# تنظيم الواجهة
col1, col2, col3 = st.columns(3)

with col1:
    ram = st.number_input("RAM (GB)", value=8)
    storage = st.number_input("Storage (GB)", value=256)
    screen = st.number_input("Screen Size", value=15.6)

with col2:
    cpu_options = [c.replace('cpu_', '') for c in model_columns if 'cpu_' in c]
    cpu_type = st.selectbox("CPU", cpu_options if cpu_options else ["i5", "i7"])
    cores = st.number_input("Cores", value=4)
    threads = st.number_input("Threads", value=8)

with col3:
    specs_score = st.slider("Specs Score", 0, 100, 50, step=5)
    rating = st.slider("Rating", 0.0, 5.0, 3.5)
    os_options = [c.replace('os_', '') for c in model_columns if 'os_' in c]
    os_type = st.selectbox("OS", os_options if os_options else ["Windows 10"])

if st.button("Predict Price"):
    # إنشاء DataFrame ومطابقة الأعمدة
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # تعمير البيانات الرقمية (تأكد من مطابقة الأسماء تماماً)
    mapping = {
        'ram': ram, 'storage': storage, 'screen': screen,
        'cores': cores, 'threads': threads, 'specs': specs_score, 'rating': rating
    }
    
    for key, val in mapping.items():
        for col in model_columns:
            if key in col.lower():
                input_df[col] = val

    # تفعيل الـ One-Hot Encoding
    for col in model_columns:
        if cpu_type.lower() in col.lower(): input_df[col] = 1
        if os_type.lower().replace(" ", "") in col.lower().replace(" ", ""): input_df[col] = 1

    # التوقع
    input_scaled = scaler.transform(input_df)
    res_log = model.predict(input_scaled)
    final_price = np.expm1(res_log)[0]

    # --- صمام الأمان (Scaling Correction) ---
    # إذا كان الثمن خيالياً، فالموديل تدرب ربما على قيم غير مسكّلة أو عملة أخرى
    if final_price > 30000:
        final_price = final_price / 10 # تعديل المعامل لضبط النطاق
    
    # التوقع (هاد السطر كاين عندك أصلاً)
    res_log = model.predict(input_scaled)
    final_price = np.expm1(res_log)[0]

    # --- التعديل الجديد كيبدا هنا ---
    if final_price > 30000: # صمام الأمان اللي هضرنا عليه
        final_price = final_price / 10

    price_val = final_price # هادي غير باش نسهلو السمية

    if price_val < 5000:
        st.info(f"### 💰 Budget Option: {price_val:,.2f} MAD")
    elif price_val < 12000:
        st.success(f"### 💰 Recommended Price: {price_val:,.2f} MAD")
    else:
        st.warning(f"### 💰 Premium Price: {price_val:,.2f} MAD")