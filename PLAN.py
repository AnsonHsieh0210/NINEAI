import os
import json
import re
import base64
import datetime
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import requests
import plotly.graph_objects as go
from googlesearch import search
from io import StringIO
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- 1. 初始化與環境設定 ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# GitHub 倉儲設定
REPO_OWNER = "AnsonHsieh0210"
REPO_NAME = "NINEAI"
FILE_PATH = "RAG.csv"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except (ValueError, TypeError) as e:
    st.error(f"Google API 金鑰設定錯誤，請檢查 .env 檔案中的 GOOGLE_API_KEY。錯誤訊息: {e}")
    st.stop()



# 安全性設定
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# 初始化模型
model = genai.GenerativeModel(
    model_name="gemini-pro", # 使用 gemini-pro 確保穩定
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)

# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "需清楚定義模型輸出，如標記位置、風險評分（0-100 分）或分類建議，指引醫師解讀結果。"},
    {"title": "介入目的", "desc": "說明臨床用途（如輔助診斷、分流）及其預期解決痛點。"},
    {"title": "警告與範圍外使用", "desc": "限制不適用情境（如特定機型、非適應症族群），並強調不得獨立作為診斷工具。"},
    {"title": "開發詳情及輸入特徵", "desc": "揭露訓練資料來源、特徵維度（如年齡、性別、影像維度等）及模型架構（如 CNN）。"},
    {"title": "確保公平性的過程", "desc": "詳述如何減少演算法偏見，確保在不同種族、性別或年齡層表現的一致性。"},
    {"title": "外部驗證過程", "desc": "展示單一中心外部驗證或跨中心聯邦驗證在真實數據表現；若為聯邦驗證須詳列中心數量及各院資料量等資訊"},
    {"title": "量化表現指標", "desc": "提供靈敏度、特異性、AUC 等具體統計數據，作為模型效能基準。"},
    {"title": "持續維護與監控", "desc": "描述部署後的技術支援、監控團隊及更新計畫，確保系統在臨床現場穩定性。"},
    {"title": "更新與持續驗證計畫", "desc": "規定再訓練頻率與定期驗證門檻，以應對醫療環境變遷的性能波動。"}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "可解釋性分析在醫療人工智慧中是指用來解釋和理解人工智慧模型如何做出預測或決策的技術和方法。這在醫療領域中至關重要，因為透明性和信任對於人工智慧工具的採用是必不可少的。其目標是提供對人工智慧系統決策過程的洞見，確保臨床醫師能夠理解和驗證其輸出結果。"},
    {"title": "AI生命週期管理", "desc": "AI 生命週期循環監測有效性在臨床醫學的應用涉及到對人工智慧（AI）系統在整個生命週期中的有效性進行持續的監測和評估。這一過程不僅包括 AI 系統的開發和部署階段，還涵蓋了後續的運行、維護和改進。這樣的監測確保了 AI系統在實際臨床環境中的表現能夠持續符合預期，並且能夠適應隨時間變化的醫療需求和資料特性，實施定期的性能監控計畫"}
]

# ---------- 3. 功能函式 ----------

def get_rag_df_from_github():
    """從 GitHub 讀取目前的 RAG 庫"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        content = base64.b64decode(res.json()['content']).decode('utf-8')
        
        # --- 核心修正處 ---
        if not content.strip():  # 如果檔案內容是空的
            return pd.DataFrame(columns=["Principle", "UserFeedback"])
        
        try:
            return pd.read_csv(StringIO(content))
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Principle", "UserFeedback"])
    
    return pd.DataFrame(columns=["Principle", "UserFeedback"])


def generalize_feedback(specific_feedback):
    # 1. 先定義 Prompt 內容
    prompt = f"""
    使用者針對醫療 AI 審查提供了具體修正建議：'{specific_feedback}'
    請將其以簡短純文字轉為通用的審查原則，使其能適用於其他不同的計畫書或不同任務模型。
    只回傳轉化後的文字，不要有其他解釋。
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "text/plain"})
    return response.text.strip()     


def update_rag_to_github(principle, feedback):
    """將回饋存入 GitHub"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    
    # 1. 取得現有資料
    df = get_rag_df_from_github()
    if "UserFeedback" not in df.columns: # 處理空檔案或格式錯誤
        df = pd.DataFrame(columns=["Principle", "UserFeedback"])

    res = requests.get(url, headers=headers)
    sha = res.json().get('sha') if res.status_code == 200 else None

    # 2. 加入新列
    new_data = pd.DataFrame([{
        "Principle": principle,
        "UserFeedback": feedback,
    }])
    df = pd.concat([df, new_data], ignore_index=True)

    # 3. 轉回 CSV 並推送到 GitHub (使用 pandas 確保格式正確)
    csv_content = df.to_csv(index=False, encoding='utf-8')
    encoded_content = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
    
    payload = {
        "message": f"Update RAG feedback for {principle}",
        "content": encoded_content,
        "sha": sha
    }
    
    put_res = requests.put(url, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

def analyze_item_with_react(item, context_text, rag_df):
    """
    使用 ReAct 模式執行單項 AI 檢核。
    這是一個代理人執行器 (Agent Executor)。
    """
    try:
        response = model.generate_content(prompt)
        clean_text = re.sub(r"```json\n?|\n?```", "", response.text).strip()
        result = json.loads(clean_text)
        # 確保額外欄位存在，若模型忘記給，則補上預設值
        result.setdefault("source", "未知")
        result.setdefault("pass_probability", 0 if result.get("status") == "不存在" else 50)
        return result
    except Exception as e:
        return {"status": "檢核錯誤", "summary": f"API 錯誤: {str(e)}", "suggestion": "", "source": "錯誤", "pass_probability": 0}
        
def get_rag_history(principle_title: str, context_text: str, rag_df: pd.DataFrame) -> str:
    """
    工具 (Tool) 1: 根據原則標題和文件內容，從 RAG 資料庫中檢索最相關的歷史經驗。
    """
    if rag_df.empty or "UserFeedback" not in rag_df.columns:
        return "RAG 知識庫是空的。"

    rel_rows = rag_df[rag_df["Principle"] == principle_title].copy()
    if rel_rows.empty:
        return f"RAG 知識庫中沒有關於 '{principle_title}' 的歷史經驗。"

    # 透過語義相似度找出最相關的歷史經驗
    try:
        pdf_context = context_text[:2000]
        pdf_vec = get_embedding(pdf_context)
        
        similarities = []
        for fb in rel_rows["UserFeedback"].tolist():
            fb_vec = get_embedding(fb)
            sim = cosine_similarity(pdf_vec, fb_vec)
            similarities.append(sim)
        
        rel_rows["sim"] = similarities
        top_3 = rel_rows.sort_values(by="sim", ascending=False).head(3)
        history = "\n".join([f"- {row['UserFeedback']}" for _, row in top_3.iterrows()])
        return f"關於 '{principle_title}' 的相關歷史經驗如下：\n{history}"
    except Exception as e:
        return f"檢索 RAG 歷史時發生錯誤: {e}"

def tool_search_web(query: str) -> str:
    """
    工具 (Tool) 2: 執行網路搜尋並返回結果摘要。
    """
    try:
        st.info(f"🔍 代理人正在執行網路搜尋: 「{query}」...")
        search_results = []
        for url in search(query, tld="com", lang="zh-TW", num=3, stop=3, pause=2):
            search_results.append(f"- {url}")
        
        if not search_results:
            return "觀察: 網路搜尋沒有找到相關結果。"
        return "觀察: 網路搜尋找到以下連結，可能包含有用資訊：\n" + "\n".join(search_results)
    except Exception as e:
        return f"觀察: 網路搜尋工具發生錯誤: {e}"

def agent_executor(item, full_text, rag_df):
    """
    ReAct 代理人執行器。
    管理 "思考 -> 行動 -> 觀察" 的循環。
    """
    # 使用一個臨時的模型來處理 ReAct 循環，因為它需要純文字來回
    agent_model = genai.GenerativeModel(model_name="gemini-pro")

    # ReAct 的核心 Prompt
    prompt_template = f"""
你是一個醫療 AI 審查代理人。你的目標是根據文件內容，評估是否符合「{item['title']}」原則。

你有以下工具可以使用：
1. `get_rag_history(principle_title: str)`: 查詢與此原則相關的歷史修正建議。
2. `tool_search_web(query: str)`: 當文件資訊不足時，上網搜尋補充資料。

你的思考與行動必須遵循以下格式：
**思考:** (你當前的分析和下一步計畫)
**行動:** (tool_name[arg]) 或 **最終答案:** (你的 JSON 格式結論)

--- 開始 ---
**任務:** 評估文件是否符合原則「{item['title']}」。
**原則定義:** {item['desc']}
**文件內容 (摘要):** {full_text[:4000]}

**思考:** 我需要評估文件是否滿足 '{item['title']}' 原則。首先，我應該檢查內部知識庫是否有相關的歷史經驗，這可以幫助我聚焦審查重點。
**行動:** get_rag_history[{item['title']}]
"""

    conversation_history = [prompt_template]
    max_turns = 5  # 防止無限循環

    for _ in range(max_turns):
        full_prompt = "\n".join(conversation_history)
        response = agent_model.generate_content(full_prompt)
        response_text = response.text.strip()
        
        conversation_history.append(response_text)

        if "**最終答案:**" in response_text:
            final_answer_part = response_text.split("**最終答案:**")[-1].strip()
            try:
                # 解析最終的 JSON 答案
                result = json.loads(re.sub(r"```json\n?|\n?```", "", final_answer_part))
                result.setdefault("pass_probability", 0 if result.get("status") == "不存在" else 50)
                return result
            except Exception as e:
                return {"status": "格式錯誤", "summary": f"無法解析最終答案: {e}", "suggestion": "", "source": "錯誤", "pass_probability": 0}
        
        elif "**行動:**" in response_text:
            action_part = response_text.split("**行動:**")[-1].strip()
            if action_part.startswith("get_rag_history"):
                observation = get_rag_history(item['title'], full_text, rag_df)
            elif action_part.startswith("tool_search_web"):
                query = action_part.split("[", 1)[1].split("]")[0]
                observation = tool_search_web(query)
            else:
                observation = "未知的行動。"
            conversation_history.append(f"**觀察:** {observation}")

    return {"status": "超時", "summary": "代理人執行超過最大輪次，未能得出結論。", "suggestion": "請檢查文件內容或簡化問題。", "source": "錯誤", "pass_probability": 0}
import numpy as np

def get_embedding(text):
    """將文字轉換為向量 - 修正模型路徑"""
    try:
        # 嘗試使用最通用的 embedding 模型名稱
        result = genai.embed_content(
            model="models/embedding-001", # 正確的 Gemini embedding 模型
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Embedding 錯誤 (雲端模式): {e}")
        return np.zeros(768)  # 回傳零向量避免後續計算崩潰
        
        

def cosine_similarity(v1, v2):
    """計算餘弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
def run_full_analysis(full_text):
    """
    執行完整的 ReAct 分析流程。
    """
    # 1. 取得歷史 RAG 資料
    rag_df = get_rag_df_from_github()

    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    results_t = []
    results_g = []

    for i, item in enumerate(all_items):
        # 為每個項目啟動一個獨立的 ReAct 代理人
        with st.status(f"代理人正在分析: {item['title']}...", expanded=False) as status:
            res = agent_executor(item, full_text, rag_df)
            status.update(label=f"分析完成: {item['title']}", state="complete")
        
        if i < 9:
            results_t.append(res)
        else:
            results_g.append(res)
            
    return {"t": results_t, "g": results_g}

def convert_results_to_csv():
    """將目前的分析結果轉換為 CSV 格式供下載"""
    if 'res_t' not in st.session_state or st.session_state['res_t'] is None:
        return None
    
    data = []
    # 處理 9 大原則
    for i, item in enumerate(st.session_state['res_t']):
        data.append({
            "分類": "九大透明性原則",
            "項目": TRANSPARENCY_9[i]['title'],
            "狀態": item['status'],
            "摘要": item['summary'],
            "建議": item['suggestion']
        })
    # 處理 2 大指標
    for i, item in enumerate(st.session_state['res_g']):
        data.append({
            "分類": "核心治理指標",
            "項目": GOVERNANCE_2[i]['title'],
            "狀態": item['status'],
            "摘要": item['summary'],
            "建議": item['suggestion']
        })
    
    df = pd.DataFrame(data)
    # 使用 StringIO 轉為 CSV 字串
    return df.to_csv(index=False).encode('utf-8-sig')

def create_gauge_chart(value, title):
    """使用 Plotly 創建儀錶板圖表"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [50, 80], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.3)'}],
        }))
    fig.update_layout(height=200, margin={'t':40, 'b':30, 'l':30, 'r':30})
    return fig

# ---------- 4. UI 介面 ----------

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統 (RAG 強化版)")

    if 'res_t' not in st.session_state:
        st.session_state['res_t'] = None

    # 在 UI 中顯示當前模式
    mode_display = "☁️ 雲端模式 (Google Gemini)"
    st.subheader(mode_display)

    with st.sidebar:
        st.header("1. 檔案讀取")
        pdf_file = st.file_uploader("上傳計畫書 PDF (選填)", type="pdf")
        st.info("若您上傳計畫書，系統將自動分析並填入部分欄位。")
        st.divider()

        st.header("2. AI 模型資訊")
        
        with st.expander("模型基本資料", expanded=True):
            st.text_input("AI模型名稱")
            st.text_input("AI Model Name")
            st.text_area("AI模型摘要", max_chars=50, help="摘要上限 50 字")
            st.text_area("AI Model Summary", max_chars=50, help="Summary max 50 characters")
            st.text_area("產品介紹", max_chars=500, help="介紹上限 500 字")
            st.text_area("Product Introduction", max_chars=500, help="Introduction max 500 characters")

        with st.expander("成效指標 (Performance Metrics)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.number_input("AUC", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                st.number_input("靈敏度 (Sensitivity)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                st.number_input("陽性預測值 (PPV)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
            with col2:
                st.number_input("準確度 (Accuracy)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                st.number_input("特異度 (Specificity)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                st.number_input("陰性預測值 (NPV)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

        with st.expander("維運計畫", expanded=True):
            st.text_area("監測計畫 (Monitoring Plan)")
            st.text_area("版本更新計畫 (Version Update Plan)")

        st.divider()
        st.header("3. 合規性分析")
        btn = st.button("🚀 開始分析", use_container_width=True)
        st.divider()
        st.caption("您的回饋建議將存入 AI 知識庫，用於強化未來分析結果。")
        st.divider()
        st.caption("本網站內容由人工智慧生成，僅為參考用途。")
        st.caption("聯絡信箱：AnsonHsieh@itri.org.tw")

    if pdf_file and btn:
        with st.spinner("正在讀取檔案並檢索歷史經驗..."):
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = "\n".join([page.get_text() for page in doc])
            results = run_full_analysis(full_text)
            st.session_state['res_t'] = results['t']
            st.session_state['res_g'] = results['g']

    # 顯示結果
    if st.session_state['res_t']:
        st.subheader("📊 九大透明性原則")
        t_data = st.session_state['res_t']
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r * 3 + c
                if idx < len(t_data):
                    item = t_data[idx]
                    with cols[c]:
                        # 創建儀錶板
                        prob = item.get('pass_probability', 0)
                        fig = create_gauge_chart(prob, f"{idx+1}. {TRANSPARENCY_9[idx]['title']}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 顯示狀態與資料來源
                        color = "green" if item['status'] == "存在" else "red"
                        source_text = item.get('source', '未知')
                        st.markdown(f"**狀態:** :{color}[{item['status']}] | **來源:** {source_text}")
                        st.info(f"**摘要:** {item['summary']}")
                        if item['suggestion']:
                            st.warning(f"💡 建議：{item['suggestion']}")

        st.divider()
        st.subheader("📋 核心治理指標")
        g_data = st.session_state['res_g']
        df_g = pd.DataFrame([{
            "評估項目": GOVERNANCE_2[i]['title'],
            "狀態": d['status'],
            "摘要": d['summary'],
            "建議": d['suggestion']
        } for i, d in enumerate(g_data)])
        st.table(df_g)
        # ---------- 新增：下載報告區塊 ----------
        st.divider()
        st.subheader("📥 匯出檢核報告")
        
        csv_data = convert_results_to_csv()
        if csv_data:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="💾 下載 CSV 報告",
                    data=csv_data,
                    file_name=f"醫療AI檢核報告_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.caption("按鈕將下載包含透明性原則與治理指標的完整彙總表格。")
                
        # 回饋收集
        st.divider()
        st.subheader("📝 訓練 AI 的判斷經驗 (RAG)")
        with st.form("rag_feedback_form"):
            all_titles = [i['title'] for i in (TRANSPARENCY_9 + GOVERNANCE_2)]
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_title = st.selectbox("選擇要修正的項目", all_titles)
            with col2:
                user_comment = st.text_area("修正建議 (AI 哪裡看錯了？)", placeholder="例如：『應加強對表格內數據的識別』或『此類資訊通常出現在附件的技術規格中』")
            submit_rag = st.form_submit_button("✅ 送出經驗並優化未來分析")

            if submit_rag:
                if not GITHUB_TOKEN:
                    st.error("請檢查 GITHUB_TOKEN 設定。")
                    st.stop()
                elif not user_comment:
                    st.warning("請填寫建議。")
                else:
                    all_results = st.session_state['res_t'] + st.session_state['res_g']
                    idx = all_titles.index(selected_title)
                    orig_sum = all_results[idx]['summary']
                    
                    with st.spinner("同步至 GitHub 中..."):
                        generalized_comment = generalize_feedback(user_comment)
                        if update_rag_to_github(selected_title, generalized_comment):
                            st.success("回饋成功！下次分析將參考此經驗。")
                        else:
                            st.error("寫入失敗，請確認 Token 權限。")

if __name__ == "__main__":
    main()
