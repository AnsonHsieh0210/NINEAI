import os
import json
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# ---------- 1. 初始化與定義 ----------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

# 使用 Gemini 1.5 Flash (兼顧速度與 JSON 解析穩定性)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# 九大透明性原則
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

# 額外兩大治理原則
GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "利用技術（如SHAP/LIME）讓醫師理解模型決策，確保輸出可驗證。"},
    {"title": "AI生命週期管理", "desc": "從開發到部署的全程風險評估與合規性監測。"}
]

# ---------- 2. 核心分析函式 ----------

def analyze_principle(principle, full_text):
    """呼叫 Gemini 進行單項檢核"""
    prompt = f"""
    你是一位專業醫療 AI 審查員。請針對以下原則檢核文件內容。
    原則名稱：{principle['title']}
    定義：{principle['desc']}
    
    文件內容：{full_text[:10000]}
    
    請以 JSON 格式回覆：
    {{
      "status": "存在" 或 "不存在",
      "summary": "若存在，請摘要說明做法；若不存在，則寫「未見相關描述」",
      "suggestion": "若不存在，請提供具體的補強建議，若存在則留空"
    }}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except:
        return {"status": "檢測失敗", "summary": "無法解析", "suggestion": "請重新執行"}

def process_analysis(full_text):
    """平行處理所有 11 項原則"""
    all_tasks = TRANSPARENCY_9 + GOVERNANCE_2
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda p: analyze_principle(p, full_text), all_tasks))
    
    return {
        "transparency": results[:9],
        "governance": results[9:]
    }

# ---------- 3. UI 呈現 (CSS & Layout) ----------

def inject_style():
    st.markdown("""
    <style>
    .flip-card { background-color: transparent; width: 100%; height: 250px; perspective: 1000px; margin-bottom: 20px; }
    .flip-card-inner { position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.6s; transform-style: preserve-3d; cursor: pointer; }
    .flip-card:hover .flip-card-inner { transform: rotateY(180deg); }
    .flip-card-front, .flip-card-back { position: absolute; width: 100%; height: 100%; -webkit-backface-visibility: hidden; backface-visibility: hidden; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 15px; border-radius: 12px; }
    .flip-card-front { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; }
    .flip-card-back { background-color: #f8fafc; color: #1e293b; transform: rotateY(180deg); border: 1px solid #e2e8f0; overflow-y: auto; text-align: left; }
    .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-top: 8px; }
    .summary-box { font-size: 0.85em; line-height: 1.4; }
    .suggest-box { font-size: 0.8em; color: #b91c1c; background: #fee2e2; padding: 5px; margin-top: 5px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config("AI Transparency Auditor", layout="wide")
    inject_style()
    
    st.title("🛡️ 負責任 AI 自動檢核系統")
    st.markdown("透過 Gemini 1.5 進行九大透明性原則與治理框架解析")

    with st.sidebar:
        st.header("文件上傳")
        uploaded_file = st.file_uploader("上傳 IRB 計畫書 (PDF)", type="pdf")
        analyze_btn = st.button("🚀 開始檢核", use_container_width=True)

    if uploaded_file and analyze_btn:
        with st.spinner("🔍 正在進行深度語義解析與合規性檢查..."):
            # PDF 解析
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            
            # 分析
            results = process_analysis(text)
            st.session_state['res_t'] = results['transparency']
            st.session_state['res_g'] = results['governance']

    # --- 輸出結果 ---
    if 'res_t' in st.session_state:
        # 第一部分：九宮格
        st.subheader("📊 九大透明性原則檢核 (Transparency)")
        t_data = st.session_state['res_t']
        
        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                item = t_data[idx]
                title = TRANSPARENCY_9[idx]['title']
                color = "#22c55e" if item['status'] == "存在" else "#ef4444"
                
                with cols[col]:
                    st.markdown(f"""
                    <div class="flip-card">
                      <div class="flip-card-inner">
                        <div class="flip-card-front">
                          <div style="font-weight:bold; font-size:1.1em;">{title}</div>
                          <div class="badge" style="background:{color};">{item['status']}</div>
                          <div style="font-size:0.7em; margin-top:15px; opacity:0.7;">🖱️ 懸停查看詳情</div>
                        </div>
                        <div class="flip-card-back">
                          <div class="summary-box"><b>摘要：</b>{item['summary']}</div>
                          {f'<div class="suggest-box"><b>建議：</b>{item["suggestion"]}</div>' if item['suggestion'] else ""}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 第二部分：治理表格
        st.divider()
        st.subheader("📋 核心治理與生命週期分析 (Governance)")
        g_data = st.session_state['res_g']
        
        # 整理成 DataFrame 顯示
        df_list = []
        for i, item in enumerate(g_data):
            df_list.append({
                "評估項目": GOVERNANCE_2[i]['title'],
                "狀態": item['status'],
                "文件摘要": item['summary'],
                "建議補充內容": item['suggestion'] if item['suggestion'] else "-(已符合)-"
            })
        
        st.table(pd.DataFrame(df_list))

        # 下載報告
        full_df = pd.concat([
            pd.DataFrame([{"Item": TRANSPARENCY_9[i]['title'], **d} for i, d in enumerate(t_data)]),
            pd.DataFrame([{"Item": GOVERNANCE_2[i]['title'], **d} for i, d in enumerate(g_data)])
        ])
        st.download_button("📥 下載完整分析報告", full_df.to_csv(index=False), "AI_Audit_Report.csv")

if __name__ == "__main__":
    main()
