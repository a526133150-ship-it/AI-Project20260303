import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 設定網頁標題與內容寬度
st.set_page_config(page_title="酒類資料集分類預測系統", page_icon="🍷", layout="wide")

# 套用全域樣式 (CSS)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #722f37; /* Wine Red */
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1 {
        color: #722f37;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine

df, wine_raw = load_data()

# --- Sidebar 左側選單 ---
st.sidebar.title("🛠️ 設定區")
selected_model = st.sidebar.selectbox(
    "請選擇預測模型：",
    ["K-Nearest Neighbors (KNN)", "Logistic Regression", "Random Forest", "XGBoost"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ 「酒類」資料集資訊")
st.sidebar.info(f"""
- **類別數量**：{len(wine_raw.target_names)} 類 ({', '.join(wine_raw.target_names)})
- **特徵數量**：{len(wine_raw.feature_names)} 項
- **樣本總數**：{len(df)}
- **資料來源**：Scikit-learn UCI Wine Dataset
""")

# --- Main 區域 ---
st.title("🍷 酒類資料集分析與預測系統")

# 分欄顯示資料預覽與統計
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 資料集前 5 筆內容")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.subheader("📊 特徵統計值資訊")
    st.dataframe(df.describe().T, use_container_width=True)

st.markdown("---")

# --- 預測區 ---
st.subheader(f"🚀 執行預測 - 當前模型：{selected_model}")

# 資料前處理與模型訓練
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if st.button("開始進行模型預測"):
    with st.spinner("模型訓練與預測中..."):
        # 選擇並訓練模型
        if selected_model == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=5)
        elif selected_model == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif selected_model == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        # 顯示結果
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="✅ 模型準確度 (Accuracy)", value=f"{acc*100:.2f}%")
        
        with res_col2:
            st.success("模型預測完成！")
            
        st.markdown("#### 🔍 測試集預測摘要 (前10筆)")
        results_df = pd.DataFrame({
            '實際類別': [wine_raw.target_names[i] for i in y_test.iloc[:10]],
            '模型預測': [wine_raw.target_names[i] for i in y_pred[:10]],
            '是否正確': ['✅' if p == a else '❌' for p, a in zip(y_pred[:10], y_test.iloc[:10])]
        })
        st.table(results_df)

st.markdown("---")
st.caption("由 Antigravity 助手開發 | 基於 Streamlit & Scikit-learn")

