import os
import json
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# ---------- 1. 初始化 ----------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ 找不到 GOOGLE_API_KEY，請檢查 .env 檔案")
    st.stop()

genai.configure(api_key=api_key)

# 建議改用 1.5 系列，穩定性與支援度最高
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config={"response_mime_type": "application/json"}
)

# ---------- 2. 定義原則 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "模型架構、訓練技術及輸出形式說明。"},
    {"title": "介入目的", "desc": "模型設計核心目標及適用情境。"},
    {"title": "警告範圍外使用", "desc": "不適用範圍及其可能發生之風險。"},
    {"title": "開發詳情及輸入", "desc": "開發過程及訓練數據特徵說明。"},
    {"title": "開發公平性過程", "desc": "防止或減輕偏見與不公平的具體方法。"},
    {"title": "外部驗證過程", "desc": "真實或模擬環境下的穩定性與泛化測試。"},
    {"title": "表現量化指標", "desc": "準確率、召回率、F1、AUC等評估指標。"},
    {"title": "實施與持續維護", "desc": "部署後的監控、修復及性能衰退處理。"},
    {"title": "更新與公平性評估", "desc": "定期重訓計畫及持續性公平性監測。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "利用技術（如SHAP/LIME）讓醫師理解模型決策。"},
    {"title": "AI生命週期管理", "desc": "從開發到部署的全程風險評估與合規性監測。"}
]

# ---------- 3. 核心功能 ----------

def clean_json_text(text):
    """防止模型回傳帶有 Markdown 標籤的 JSON"""
    return re.sub(r"```json\n?|\n?```", "", text).strip()

def analyze_principle(principle, full_text):
    prompt = f"""
    請擔任醫療 AI 審查員。檢核文件是否符合：{principle['title']}
    定義：{principle['desc']}
    文件內容：{full_text[:10000]}
    請回覆 JSON: {{"status": "存在/不存在", "summary": "摘要", "suggestion": "建議"}}
    """
    try:
        response = model.generate_content(prompt)
        # 增加安全性解析
        clean_text = clean_json_text(response.text)
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "檢測失敗", "summary": f"錯誤：{str(e)}", "suggestion": "請手動檢查"}

def process_analysis(full_text):
    all_tasks = TRANSPARENCY_9 + GOVERNANCE_2
    # 使用 ThreadPool 提升速度
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda p: analyze_principle(p, full_text), all_tasks))
    return {"transparency": results[:9], "governance": results[9:]}

# ---------- 4. UI 呈現 ----------
def main():
    st.set_page_config(page_title="AI 負責任檢核", layout="wide")
    st.title("🛡️ 醫療 AI 透明性與治理檢核")

    with st.sidebar:
        uploaded_file = st.file_uploader("上傳計畫書 (PDF)", type="pdf")
        analyze_btn = st.button("🚀 開始執行分析")

    if uploaded_file and analyze_btn:
        with st.spinner("分析中..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            
            results = process_analysis(text)
            st.session_state['res_t'] = results['transparency']
            st.session_state['res_g'] = results['governance']

    if 'res_t' in st.session_state:
        # 九宮格渲染 (簡化版)
        st.subheader("九大透明性原則")
        t_data = st.session_state['res_t']
        cols = st.columns(3)
        for i, item in enumerate(t_data):
            with cols[i % 3]:
                color = "green" if item['status'] == "存在" else "red"
                st.info(f"**{TRANSPARENCY_9[i]['title']}**")
                st.markdown(f"狀態：:{color}[{item['status']}]")
                with st.expander("查看摘要"):
                    st.write(item['summary'])
                    if item['suggestion']:
                        st.warning(f"建議：{item['suggestion']}")

        st.divider()
        st.subheader("核心治理分析")
        st.table(pd.DataFrame([{
            "項目": GOVERNANCE_2[i]['title'],
            **d
        } for i, d in enumerate(st.session_state['res_g'])]))

if __name__ == "__main__":
    main()
