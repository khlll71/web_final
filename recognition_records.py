# recognition_records.py

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime


# 資料庫連線


def display_recognition_records():
    conn = sqlite3.connect('recognition_records.db')
    cursor = conn.cursor()
    # 顯示瀏覽紀錄
    cursor.execute('SELECT result,date FROM recognition_records')
    data = cursor.fetchall()

    # 將查詢結果轉換為Pandas DataFrame
    df = pd.DataFrame(data, columns=['result', 'date'])
    st.dataframe(df,width=1800,height=500)
    conn.close()


def save(result):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('recognition_records.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO recognition_records (result, date) VALUES (?,? )', ( result, date,))
    conn.commit()

 
