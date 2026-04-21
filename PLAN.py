import os
import json
import re
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("請在 .env 檔案中設定 GOOGLE_API_KEY")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 設定安全性：放寬醫療專有名詞的限制，確保解析不中斷
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化 Gemini 2.5 Pro
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro", # 根據您的 API 權限設定
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2,  # 降低隨機性，提高分析穩定度
    },
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "需清楚定義產出的具體內容，如標記位置、風險評分（0-100 分）或分類建議，指引醫師解讀結果。"},
    {"title": "介入目的", "desc": "說明 AI 的臨床用途（如輔助診斷、分流或篩查）及其預期解決的具體臨床痛點。"},
    {"title": "警告與範圍外使用", "desc": "明確限制條件，告知醫師不適用情境（如特定機型、非適應症族群），並強調不得獨立作為診斷工具。"},
    {"title": "開發詳情及輸入特徵", "desc": "揭露訓練資料來源、特徵維度（如年齡、像素、密度、腫塊）及採用的算法架構（如 CNN）。"},
    {"title": "確保公平性的過程", "desc": "詳述如何檢查並減少演算法偏見，確保在不同種族、性別或年齡層表現的一致性。"},
    {"title": "外部驗證過程", "desc": "展示在獨立真實世界數據上的表現，包含跨中心數量、硬體製造商分佈及組織學類型。"},
    {"title": "量化表現指標", "desc": "提供靈敏度、特異性、AUC 等具體統計數據，作為醫師評估系統效能的基準。"},
    {"title": "持續維護與監控", "desc": "描述部署後的技術支援、監控團隊及更新計畫，確保系統在臨床現場的穩定性。"},
    {"title": "更新與持續驗證計畫", "desc": "規定再訓練頻率與定期驗證門檻，以應對醫療環境變遷導致的性能波動。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "可解釋性分析在醫療人工智慧中是指用來解釋和理解人工智慧模型如何做出預測或決策的技術和方法。這在醫療領域中至關重要，因為透明性和信任對於人工智慧工具的採用是必不可少的。其目標是提供對人工智慧系統決策過程的洞見，確保臨床醫師能夠理解和驗證其輸出結果。"},
    {"title": "AI生命週期管理", "desc": "AI 生命週期循環監測有效性在臨床醫學的應用涉及到對人工智慧（AI）系統在整個生命週期中的有效性進行持續的監測和評估。這一過程不僅包括 AI 系統的開發和部署階段，還涵蓋了後續的運行、維護和改進。這樣的監測確保了 AI系統在實際臨床環境中的表現能夠持續符合預期，並且能夠適應隨時間變化的醫療需求和資料特性，實施定期的性能監控計畫"}
]

# ---------- 3. 核心邏輯 ----------

def analyze_item(item, context_text):
    """呼叫 Gemini 2.5 Pro 進行單項分析"""
    prompt = f"""
    你是一位醫療 AI 合規性審查專家。請針對以下原則分析文件內容：
    原則：{item['title']}
    定義：{item['desc']}
    
    文件內容：{context_text[:12000]}
    
    請以 JSON 格式回覆，不需其他解釋：
    {{
      "status": "存在" 或 "不存在",
      "summary": "具體做法摘要（若不存在則寫未見描述）",
      "suggestion": "缺失建議（若存在則留空）"
    }}
    """
    try:
        response = model.generate_content(prompt)
        # 清除 Markdown 標籤以防解析錯誤
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"status": "檢核錯誤", "summary": f"API 錯誤: {str(e)}", "suggestion": ""}

def run_full_analysis(full_text):
    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    # 2.5 Pro 支援高併發，這裡開 6 個 Thread 同時跑
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda x: analyze_item(x, full_text), all_items))
    return {"t": results[:9], "g": results[9:]}

# ---------- 4. UI 介面 (Streamlit) ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統 (Gemini 2.5 Pro)")

    with st.sidebar:
        st.header("1. 檔案讀取")
        pdf_file = st.file_uploader("上傳計畫書 PDF", type="pdf")
        btn = st.button("🚀 開始分析", use_container_width=True)

    if pdf_file and btn:
        with st.spinner("Gemini 2.5 Pro 正在分析中..."):
            # 讀取 PDF 文字
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            
            # 分析
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # --- 顯示結果 ---
    if 'res_t' in st.session_state:
        st.subheader("📊 九大透明性原則 (九宮格)")
        t_data = st.session_state['res_t']
        
        # 3x3 呈現
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                item = t_data[idx]
                with cols[c]:
                    color = "green" if item['status'] == "存在" else "red"
                    st.markdown(f"### {idx+1}. {TRANSPARENCY_9[idx]['title']}")
                    st.markdown(f"**狀態：** :{color}[{item['status']}]")
                    st.info(item['summary'])
                    if item['suggestion']:
                        st.warning(f"💡 建議：{item['suggestion']}")

        st.divider()
        st.subheader("📋 核心治理指標 (表格)")
        g_data = st.session_state['res_g']
        df_g = pd.DataFrame([{
            "評估項目": GOVERNANCE_2[i]['title'],
            "狀態": d['status'],
            "摘要": d['summary'],
            "建議": d['suggestion']
        } for i, d in enumerate(g_data)])
        st.table(df_g)

if __name__ == "__main__":
    main()
