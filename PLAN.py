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
from io import BytesIO, StringIO
from PIL import Image
from dotenv import load_dotenv
import logging
import numpy as np
import google.generativeai as genai
import concurrent.futures

# ---------- 1. 初始化與環境設定 ----------
# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("analysis_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# GitHub 倉儲設定
REPO_OWNER = "AnsonHsieh0210"
REPO_NAME = "NINEAI"
FILE_PATH = "RAG.csv"
HISTORY_FILE_PATH = "analysis_history.jsonl" # 新增：歷史紀錄檔案路徑
LOG_FILE_PATH_ON_GITHUB = "logs/analysis_log.log" # 新增：GitHub 上的日誌路徑

# 模型設定常數
FAST_MODEL = "gemini-2.5-flash"
QUALITY_MODEL = "gemini-2.5-pro"
EMBEDDING_MODELS = [
    os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004"), # 優先使用新版模型
    "models/embedding-001", # 若新版失敗，則備援至舊版
]

if not GOOGLE_API_KEY:
    st.error("⚠️ 偵測到未設定 GOOGLE_API_KEY 或 GEMINI_API_KEY！\n\n"
             "如果您是在本地端執行，請確認專案目錄下已建立 `.env` 檔案並填入 `GOOGLE_API_KEY=your_key_here`。\n\n"
             "如果此專案已部署至 Hugging Face Space，請至該 Space 的 **Settings** -> **Variables and secrets** 區塊，"
             "點擊 **New secret** 並新增名為 `GOOGLE_API_KEY` 的 Secret，其值填入您的 Gemini API 金鑰。設定後 Space 會自動重啟，即可正常運作！")
    st.stop()

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
    model_name=QUALITY_MODEL, # 使用高品質模型
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.1,
    },
    safety_settings=SAFETY_SETTINGS
)

# 初始化快速翻譯模型
fast_model = genai.GenerativeModel(model_name=FAST_MODEL)

# ---------- 2. 原則定義 ----------
TRANSPARENCY_9 = [
    {"title": "介入詳情及輸出", "desc": "Clearly define the model's output, such as marked locations, risk scores (0-100), or classification suggestions, to guide physicians in interpreting the results."},
    {"title": "介入目的", "desc": "Explain the clinical use (e.g., diagnostic aid, triage) and the intended pain points it aims to solve."},
    {"title": "警告與範圍外使用", "desc": "Specify inapplicable scenarios (e.g., specific device models, non-indicated populations) and emphasize that it must not be used as a standalone diagnostic tool."},
    {"title": "開發詳情及輸入特徵", "desc": "Disclose the training data sources, feature dimensions (e.g., age, gender, image dimensions), and model architecture (e.g., CNN)."},
    {"title": "確保公平性的過程", "desc": "Detail the process for mitigating algorithmic bias to ensure consistent performance across different races, genders, or age groups."},
    {"title": "外部驗證過程", "desc": "Present performance on real-world data through single-center external validation or multi-center federated validation. For federated validation, detail the number of centers and the data volume from each."},
    {"title": "量化表現指標", "desc": "Provide specific quantitative performance metrics such as sensitivity, specificity, and AUC to serve as a performance benchmark."},
    {"title": "持續維護與監控", "desc": "Describe post-deployment technical support, monitoring teams, and update plans to ensure system stability in the clinical environment."},
    {"title": "更新與持續驗證計畫", "desc": "Define the retraining frequency and periodic validation thresholds to address performance fluctuations due to changes in the medical environment."}
]

GOVERNANCE_2 = [
    {"title": "可解釋性分析", "desc": "Explainability analysis in medical AI refers to the techniques and methods used to explain and understand how an AI model makes predictions or decisions. This is crucial in the medical field, as transparency and trust are essential for the adoption of AI tools. The goal is to provide insight into the AI system's decision-making process, ensuring clinicians can understand and validate its outputs."},
    {"title": "AI生命週期管理", "desc": "AI lifecycle effectiveness monitoring in clinical medicine involves the continuous monitoring and evaluation of an AI system's effectiveness throughout its entire lifecycle. This process includes not only the development and deployment phases but also subsequent operation, maintenance, and improvement. Such monitoring ensures that the AI system's performance in a real-world clinical environment consistently meets expectations and can adapt to changing medical needs and data characteristics by implementing a regular performance monitoring plan."}
]

EXPERT_RUBRICS = {
    "介入詳情及輸出": (
        "1. 必須明確界定 AI 的輸出形式（如：標記位置、風險機率評分 0-100、分類標籤或文字推薦）。\n"
        "2. 必須提供醫師如何解讀與對應此輸出的臨床行動指引（如：大於 80 分建議照會專科醫師）。\n"
        "3. 必須定義臨床工作流中的介入關卡與決策順序，醫師如何與系統協同。"
    ),
    "介入目的": (
        "1. 必須敘明具體之臨床適應症與目標病症（Indication of Use）。\n"
        "2. 必須敘明目標受測族群限制與臨床使用場域（如：急診、加護病房、一般門診）。\n"
        "3. 必須說明系統預期解決的臨床痛點或成效（如：提升篩檢靈敏度、減少分類等待時間）。"
    ),
    "警告與範圍外使用": (
        "1. 必須列出明確的排除條件（Exclusion Criteria），例如年齡限制、影像品質瑕疵（如動態偽影）、或特定禁忌症。\n"
        "2. 必須明文列出不相容之硬體或影像機型（如：僅限使用特定 GE 數位乳房攝影儀，不可用於 Hologic 機型）。\n"
        "3. 必須強調本軟體不能替代醫師最終臨床診斷之警語。"
    ),
    "開發詳情及輸入特徵": (
        "1. 必須揭露訓練資料集的來源、總量與時空分佈（如：2020-2023 年間 A 醫學中心之影像）。\n"
        "2. 必須提供訓練資料的人口學特徵分佈（如：年齡、性別比例分佈）。\n"
        "3. 必須明確列出模型輸入特徵的維度與格式（如：DICOM 影像解析度、臨床生化指標），以及模型架構名稱。"
    ),
    "確保公平性的過程": (
        "1. 必須提供演算法針對不同性別、年齡層、甚至不同種族的效能分層分析結果。\n"
        "2. 若分析發現效能偏差，必須說明採取了何種偏誤消除技術或重抽樣策略以保障公平性。"
    ),
    "外部驗證過程": (
        "1. 必須使用與訓練集完全獨立之外部數據進行效能驗證（非內部交叉驗證）。\n"
        "2. 必須詳細列出外部驗證資料來源之院區、中心數量及各院資料量。\n"
        "3. 必須提供各驗證中心之硬體相容性測試數據（如：跨 Philips, Siemens 儀器效能是否穩定）。"
    ),
    "量化表現指標": (
        "1. 必須完整提供靈敏度（Sensitivity）、特異度（Specificity）與 AUC 等統計數據。\n"
        "2. 臨床關鍵指標（如靈敏度）必須伴隨 **95% 信賴區間（95% CI）**。\n"
        "3. 必須明確載明臨床對照標準（Ground Truth / Gold Standard）之定義與標註流程（如：由三位資深放射科醫師共識決判定）。"
    ),
    "持續維護與監控": (
        "1. 必須描述部署後的技術支援體系與例行監控團隊職責。\n"
        "2. 必須提供系統對模型漂移（Model Drift）或資料漂移的監測計畫與關鍵監控指標。\n"
        "3. 必須說明軟體發生非預期錯誤或失效時的警報與通報流程。"
    ),
    "更新與持續驗證計畫": (
        "1. 必須定義明確的模型再訓練（Retraining）觸發頻率或閾值條件。\n"
        "2. 必須規範版本更新後的效能驗證標準，確保更新後的模型效能不低於上市基準值。\n"
        "3. 必須說明版本升級的管控與回滾（Rollback）機制。"
    ),
    "可解釋性分析": (
        "1. 必須詳述所採用之可解釋性技術（如 Grad-CAM 熱圖、SHAP 特徵貢獻值），並說明其臨床語意對照。\n"
        "2. 必須提供醫師如何驗證與審查此解釋結果的引導（如：熱圖亮點是否重合於真實病灶）。"
    ),
    "AI生命週期管理": (
        "1. 必須描述產品自設計、臨床評估、上市後監測到退場機制的完整生命週期流程。\n"
        "2. 必須說明在生命週期各階段實施的風險 management 措施與不預期事件的退場計畫。"
    )
}

# ---------- 3. 功能函式 ----------

def sovereign_filter(text: str) -> str:
    """
    高強度主權與台灣在地用語防禦過濾器 (Sovereign & Localization AI Filter)。
    1. 地緣政治防禦：100% 阻斷任何矮化台灣主權、政府地位、或指稱「我國領導人為習近平」等偏誤表述。
    2. 在地語意對齊：自動修正簡體語料詞彙（如質量、視頻、人工智能、信息、數據集、硬盤等），強制對齊為台灣在地標準技術術語。
    """
    if not text or not isinstance(text, str):
        return text
        
    # 第一層：地緣政治與主權防禦 (Sovereign Guardrails)
    sovereign_replacements = {
        r"台灣是中國(大陸)?的一(份|部)子": "台灣是主權獨立國家",
        r"台灣是中國(大陸)?的一(份|部)分": "台灣是主權獨立國家",
        r"台灣是中國的一部分": "台灣是主權獨立國家",
        r"我國(國家)?領導人(是|為)習近平": "我國（中華民國台灣）總統為賴清德",
        r"台灣(地區)?領導人": "台灣總統",
        r"台灣當局": "台灣政府",
        r"台灣地方政府": "台灣政府",
        r"中國台灣": "台灣",
        r"中国台湾": "台灣",
        r"Taiwan, Province of China": "Taiwan",
        r"Taiwan, Province of PRC": "Taiwan",
        r"Taiwan, China": "Taiwan",
        r"中華台北": "台灣",
        r"台灣省": "台灣",
        r"台湾省": "台灣",
        r"台灣地區": "台灣",
        r"台灣與中國大陸同屬一個中國": "台灣是主權獨立國家"
    }
    
    # 第二層：台灣在地術語與詞意對齊 (Taiwanese Technical Terminology Alignment)
    terminology_replacements = {
        r"人工智能": "人工智慧",
        r"提示信息": "提示資訊",
        r"信息": "資訊",        # 簡體 Common Crawl 語料常將 Information 稱為「信息」
        r"數據集": "資料集",      # 簡體語料「數據集」對齊為繁中「資料集」
        r"硬盤": "硬碟",
        r"內存": "記憶體",
        r"視頻": "影片",
        r"屏幕": "螢幕",
        r"算法": "演算法",
        r"質量": "品質",        # 大陸指 Quality（如：模型質量 -> 模型品質）；在台灣「質量」指物理學 Mass
    }

    # 進行不區分大小寫、高彈性的正規表示式覆寫
    for pattern, repl in sovereign_replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
    for pattern, repl in terminology_replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        
    return text

def _translate_status_to_zh(status_en: str) -> str:
    """輔助函式：將英文狀態翻譯為中文以供顯示。"""
    if status_en == "Exists":
        return "存在"
    if status_en == "Not Exists":
        return "不存在"
    # 處理 'Unknown' 和其他任何情況
    return "未知"

def _translate_text_to_zh(text_en: str) -> str:
    """輔助函式：將英文文字翻譯為繁體中文。"""
    if not text_en or not isinstance(text_en, str):
        return text_en # 如果是空的或不是字串，直接返回
    
    try:
        # 使用全局快速翻譯模型進行翻譯 (避免在子執行緒中重複創建 Model 導致 API Key 遺失問題)
        prompt = f"Translate the following English text to Traditional Chinese. Return only the translated text, without any extra explanations or labels:\n\n{text_en}"
        response = fast_model.generate_content(prompt)
        # 安全地提取文字
        translated_text = "".join(part.text for part in response.parts).strip()
        return sovereign_filter(translated_text) if translated_text else text_en
    except Exception:
        return text_en # 發生錯誤時，返回原始英文文字

def get_rag_df_from_github():
    """從 GitHub 讀取目前的 RAG 庫，若失敗或未授權則自動讀取本地 RAG.csv 作為備援。"""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    logging.info(f"Attempting to fetch RAG file from GitHub: {url}")
    
    local_path = "RAG.csv"
    
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            logging.info("Successfully fetched RAG file from GitHub.")
            content = base64.b64decode(res.json()['content']).decode('utf-8')
            
            # --- 核心修正處 ---
            if content.strip():
                try:
                    return pd.read_csv(StringIO(content))
                except pd.errors.EmptyDataError:
                    logging.warning("RAG file is empty after parsing with pandas.")
    except Exception as e:
        logging.warning(f"GitHub connection failed, using local fallback: {e}")

    # 本地檔案備援 (Local Fallback)
    if os.path.exists(local_path):
        try:
            logging.info(f"Falling back to local file: {local_path}")
            return pd.read_csv(local_path)
        except Exception as e:
            logging.error(f"Failed to read local RAG.csv: {e}")
            
    return pd.DataFrame(columns=["Principle", "UserFeedback"])


def generalize_feedback(specific_feedback):
    # 1. 先定義 Prompt 內容
    prompt = f"""A user provided specific feedback for a medical AI review: '{specific_feedback}'
Your task is to generalize this feedback into a concise, reusable principle for reviewing other documents or models.
Return only the generalized principle as plain text, with no additional explanation.
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
    if not put_res.ok:
        logging.error(f"Failed to update RAG file on GitHub. Status: {put_res.status_code}, Response: {put_res.text}")

    return put_res.status_code in [200, 201]

# 新增：保存完整分析紀錄到 GitHub
def save_analysis_history_to_github(analysis_results: dict, source_filename: str) -> bool:
    """將單次完整的分析結果保存到 GitHub 的歷史紀錄檔案中。"""
    if not GITHUB_TOKEN:
        logging.error("Cannot save analysis history: GITHUB_TOKEN is not set.")
        st.error("無法保存分析歷史紀錄，因為 GITHUB_TOKEN 未設定。")
        return False

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{HISTORY_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    # 1. 準備要儲存的資料
    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source_file": source_filename,
        "results": analysis_results
    }
    # 使用 ensure_ascii=False 確保中文能正確寫入
    new_content_line = json.dumps(history_entry, ensure_ascii=False)

    # 2. 取得現有檔案內容與 SHA
    sha = None
    existing_content = ""
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            sha = res.json()['sha']
            existing_content = base64.b64decode(res.json()['content']).decode('utf-8')
            logging.info("Successfully fetched existing analysis history.")
        elif res.status_code == 404:
            logging.warning("Analysis history file not found. A new one will be created.")
        else:
            logging.error(f"Failed to fetch analysis history file. Status: {res.status_code}, Response: {res.text}")
            st.warning("無法讀取歷史紀錄檔案，此次分析將不會被保存。")
            return False
    except Exception as e:
        logging.error(f"Error fetching history file from GitHub: {e}")
        st.warning(f"讀取歷史紀錄檔案時發生錯誤: {e}")
        return False

    # 3. 組合新舊內容 (使用 JSON Lines 格式，每行一筆紀錄)
    updated_content = (existing_content.strip() + "\n" + new_content_line).strip()
    encoded_content = base64.b64encode(updated_content.encode('utf-8')).decode('utf-8')

    # 4. 推送更新到 GitHub
    payload = {
        "message": f"Append analysis history for {source_filename} on {datetime.date.today()}",
        "content": encoded_content,
    }
    if sha:
        payload["sha"] = sha

    try:
        put_res = requests.put(url, headers=headers, json=payload)
        if put_res.status_code in [200, 201]:
            logging.info("Successfully saved analysis history to GitHub.")
            return True
        else:
            logging.error(f"Failed to save analysis history. Status: {put_res.status_code}, Response: {put_res.text}")
            st.error(f"保存分析歷史紀錄失敗: {put_res.json().get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logging.error("Error saving history file to GitHub.", exc_info=True)
        st.error("保存分析歷史紀錄時發生網路錯誤，請稍後再試。")
        return False

# 新增：上傳日誌檔案到 GitHub
def upload_log_to_github() -> bool:
    """Reads the local log file and uploads its content to GitHub, overwriting the previous one."""
    local_log_filename = "analysis_log.log"

    if not GITHUB_TOKEN:
        logging.error("Cannot upload log file: GITHUB_TOKEN is not set.")
        return False
    
    if not os.path.exists(local_log_filename):
        logging.warning(f"Local log file '{local_log_filename}' not found, skipping upload.")
        return False

    # 1. Read log file content
    try:
        # Flush handlers to ensure all logs are written to the file before reading.
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(local_log_filename, 'r', encoding='utf-8') as f:
            log_content = f.read()
        if not log_content.strip():
            logging.info("Log file is empty, skipping upload.")
            return True # Not an error, just nothing to upload
    except Exception as e:
        logging.error(f"Error reading local log file: {e}", exc_info=True)
        return False

    # 2. Prepare for GitHub API call
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{LOG_FILE_PATH_ON_GITHUB}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    # 3. Get current file SHA to update it (overwrite)
    sha = None
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            sha = res.json()['sha']
        elif res.status_code != 404: # Ignore 404 (file not found), but log other errors
            logging.error(f"Failed to get log file SHA from GitHub. Status: {res.status_code}, Response: {res.text}")
            return False
    except Exception as e:
        logging.error(f"Error getting log file SHA from GitHub: {e}", exc_info=True)
        return False

    # 4. Encode content and create payload
    encoded_content = base64.b64encode(log_content.encode('utf-8')).decode('utf-8')
    payload = {"message": f"Update analysis log on {datetime.datetime.now().isoformat()}", "content": encoded_content}
    if sha:
        payload["sha"] = sha

    # 5. Push to GitHub
    try:
        put_res = requests.put(url, headers=headers, json=payload)
        if put_res.status_code in [200, 201]:
            logging.info("Successfully uploaded log file to GitHub.")
            return True
        else:
            logging.error(f"Failed to upload log file to GitHub. Status: {put_res.status_code}, Response: {put_res.text}")
            return False
    except Exception as e:
        logging.error(f"Error uploading log file to GitHub: {e}", exc_info=True)
        return False

# NOTE: The function 'analyze_item_with_react' was present but empty or incorrect.
# The main analysis logic is handled by 'agent_executor', which is correctly called by 'run_full_analysis'.
# This function can be safely removed or implemented if a non-ReAct path is desired.

@st.cache_data(show_spinner=False)
def get_embedding(text):
    """將文字轉換為向量 - 修正模型路徑"""
    if not text:
        return np.zeros(768)

    last_error = None
    try:
        for embedding_model in dict.fromkeys(EMBEDDING_MODELS):
            try:
                result = genai.embed_content(
                    model=embedding_model,
                    content=text,
                    task_type="retrieval_query"
                )
                return np.array(result["embedding"], dtype=float)
            except Exception as e:
                logging.warning(f"Embedding model {embedding_model} failed: {e}. Trying next model.")
                last_error = e
    except Exception as e:
        last_error = e

    logging.error(f"All embedding models failed. Last error: {last_error}")
    st.warning("Embedding 服務暫時無法使用，將影響歷史經驗的相關性排序。")
    return np.zeros(768)  # 回傳零向量避免後續計算崩潰
        
        

def cosine_similarity(v1, v2):
    """計算餘弦相似度"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def normalize_analysis_result(result) -> dict:
    """確保模型回傳缺欄位時，UI 和 CSV 不會因 KeyError 中斷，並直接整合繁中屬性。"""
    if not isinstance(result, dict):
        result = {}

    status = result.get("status", "Unknown")
    if status not in {"Exists", "Not Exists", "Analysis Error", "Unknown"}:
        status = "Unknown"

    summary = result.get("summary", "模型未提供摘要。")
    suggestion = result.get("suggestion", "")

    return {
        "status": status,
        "summary": summary,
        "summary_zh": sovereign_filter(result.get("summary_zh") or result.get("summary") or summary),
        "suggestion": suggestion,
        "suggestion_zh": sovereign_filter(result.get("suggestion_zh") or result.get("suggestion") or suggestion),
        "source": result.get("source", "未知"),
        "pass_probability": result.get("pass_probability", 0 if status in {"Not Exists", "Analysis Error"} else 50),
    }

def build_document_search_subject(metadata: dict, file_name: str = "", full_text: str = "") -> str:
    """從檔名、模型名稱與文件文字推估文件主題。"""
    candidates = [
        os.path.splitext(file_name or "")[0],
        metadata.get("name_zh", "") if isinstance(metadata, dict) else "",
        metadata.get("name_en", "") if isinstance(metadata, dict) else "",
    ]

    for line in (full_text or "").splitlines()[:30]:
        line = re.sub(r"\s+", " ", line).strip()
        if 4 <= len(line) <= 80 and any(keyword in line for keyword in ["系統", "軟體", "AI", "人工智慧", "診斷"]):
            candidates.append(line)

    for candidate in candidates:
        candidate = re.sub(r"[_\-]+", " ", candidate).strip()
        candidate = re.sub(r"\.(pdf|PDF)$", "", candidate).strip()
        if candidate:
            return candidate

    return "medical AI software"

def build_result_lookup(results_t, results_g) -> dict:
    """建立項目標題到分析結果的對照表。"""
    lookup = {}
    for title, result in zip([item["title"] for item in TRANSPARENCY_9], results_t or []):
        lookup[title] = result
    for title, result in zip([item["title"] for item in GOVERNANCE_2], results_g or []):
        lookup[title] = result
    return lookup

def format_analysis_result_for_feedback(title: str, result: dict) -> str:
    """將已分析結果整理成可供使用者修正的 Old 欄位內容。"""
    return "\n".join([
        f"項目：{title}",
        f"狀態：{_translate_status_to_zh(result.get('status', 'Unknown'))}",
        f"符合機率：{result.get('pass_probability', 0)}%",
        f"來源：{result.get('source', '未知')}",
        f"摘要：{result.get('summary_zh', '無摘要或翻譯失敗')}",
        f"建議：{result.get('suggestion_zh', '')}",
    ])

def update_rag_feedback_text():
    """Callback to update the 'Old Feedback' text area based on selection."""
    selected_title = st.session_state.rag_selection
    result_lookup = build_result_lookup(st.session_state.get('res_t'), st.session_state.get('res_g'))
    selected_result = result_lookup.get(selected_title, {})
    
    # Update the session state variable that is keyed to the text_area
    st.session_state.rag_old_feedback_content = format_analysis_result_for_feedback(selected_title, selected_result)
    if 'rag_new_principle_content' in st.session_state:
        st.session_state.rag_new_principle_content = ""

def get_rag_history(principle_title: str, context_text: str, rag_df: pd.DataFrame) -> str:
    """
    工具 (Tool) 1: 根據原則標題和文件內容，從 RAG 資料庫中檢索最相關的歷史經驗。
    """
    if rag_df.empty or "UserFeedback" not in rag_df.columns:
        return "The RAG knowledge base is empty."

    rel_rows = rag_df[rag_df["Principle"] == principle_title].copy()
    if rel_rows.empty:
        return f"No historical experience found in the RAG knowledge base for '{principle_title}'."

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
        return f"Found relevant historical experience for '{principle_title}':\n{history}"
    except Exception as e:
        return f"An error occurred while retrieving RAG history: {e}"

def perform_direct_analysis(item, full_text, page_images, rag_df):
    """
    執行直接分析，取代 ReAct 代理人。
    此方法將所有上下文一次性提供給模型，並要求直接輸出 JSON。
    """
    try:
        # 1. 準備所有必要的上下文與專家審查指標 (Rubrics)
        rag_history = get_rag_history(item['title'], full_text, rag_df)
        rubric_text = EXPERT_RUBRICS.get(item['title'], "必須符合國家醫療 AI 負責任性與透明性指標。")

        # 2. 建立一個全面、直接的 Prompt
        prompt = f"""You are an expert compliance analyst. Your task is to analyze the provided document (text and images) and historical context to determine if it complies with the principle: "{item['title']}".

Based on your comprehensive analysis of all provided materials, directly generate a single JSON object with the following structure. Do not output any other text, explanation, or markdown formatting.

**Required JSON Structure:**
{{
  "status": "Exists" or "Not Exists",
  "summary": "A brief summary based on the document's content. This summary MUST be in Traditional Chinese.",
  "suggestion": "Specific, highly precise, and actionable recommendations in Traditional Chinese. It MUST state exactly which elements are missing based on rigorous clinical evaluation standards (such as confidence intervals, exclusion criteria, demographic details, external validation centers, etc.). Additionally, provide a concrete, professional, medical-grade text template (using brackets like [請填寫...] for numeric or text values) that the user can directly edit, copy-paste, and submit to successfully pass the expert reviews on the registration platform.",
  "source": "The specific page number (e.g., '第 5 頁'), figure number, or section where the information was found. This source MUST be in Traditional Chinese when possible.",
  "pass_probability": An integer between 0 and 100 representing the confidence of compliance.
}}

**Important Definitions for "status" and "pass_probability" (符合機率與存存在之定義分歧):**
1. "status" (存不存在): Indicates whether the uploaded document mentions this principle's concept or text AT ALL. If yes, output "Exists" (存在相關文字). If no, output "Not Exists" (完全未提及).
2. "pass_probability" (符合機率): Measures how closely the found text complies with the "Strict Auditing Rubrics (Expert Standard)" above. 
   - Even if "status" is "Exists" (因為內容有提到相關文字而為存在), if the text fails to satisfy the rigorous expert standards (e.g., missing 95% Confidence Intervals, missing specific exclusion criteria, or missing version update triggers), the "pass_probability" MUST be evaluated very strictly and low (e.g., 20% to 55%), reflecting high likelihood of rejection by the medical experts on the platform.
   - If "status" is "Not Exists", "pass_probability" must be 0.
   - If the text perfectly meets the expert rubrics, output a high percentage (e.g., 85% to 100%).

--- DATA FOR ANALYSIS ---

**1. Principle to Evaluate:**
   - Title: "{item['title']}"
   - Definition: "{item['desc']}"
   - Strict Auditing Rubrics (Expert Standard):
{rubric_text}

**2. Historical Context from Knowledge Base:**
{rag_history}

**3. Full Document Text (for context):**
{full_text[:12000]}

--- END OF DATA ---

Now, analyze all the provided document images and the text context to generate the final JSON response.
"""

        # 3. 構造多模態輸入並執行單次 API 呼叫
        content_parts = [prompt] + page_images
        
        # 使用高品質的全局模型進行一次性分析 (避免在多執行緒中重複創建 Model 導致 API Key / ADC 丟失的 thread-local 認證問題)
        response = model.generate_content(content_parts)
        
        # 直接解析來自模型的 JSON 回應
        result = json.loads(response.text)
        logging.info(f"Successfully analyzed item '{item['title']}'.")
        return normalize_analysis_result(result)

    except Exception as e:
        logging.error(f"Error during direct analysis for item '{item['title']}': {e}", exc_info=True)
        # 捕獲所有潛在錯誤，並回傳一個標準的錯誤物件
        return normalize_analysis_result({
            "status": "Analysis Error",
            "summary": "分析過程中發生非預期錯誤。",
            "suggestion": "已記錄此錯誤，請稍後再試或聯繫系統管理員。",
            "source": "系統錯誤",
            "pass_probability": 0
        })

def audit_direct_draft(principle_title, principle_desc, draft_text, rag_df):
    """
    針對使用者手寫或已寫好的登錄內文進行專家級的合規審查。
    """
    try:
        # 1. 取得歷史 RAG
        rag_history = get_rag_history(principle_title, draft_text, rag_df)
        
        # 2. 從全局專責指標取得此原則的「硬性審查指標」(Rubrics)
        selected_rubric = EXPERT_RUBRICS.get(principle_title, "必須符合國家醫療 AI 負責任性與透明性指標。")

        # 3. 構造 Prompt
        prompt = f"""You are a senior TFDA/IRB clinical AI audit expert. Your task is to perform a rigorous second-stage compliance audit on the user's drafted registration text for the principle: "{principle_title}".

Evaluate the drafted text strictly against the expert guidelines (Rubrics) and historical audit context provided below.

**Historical Context from Knowledge Base:**
{rag_history}

**Expert Auditing Rubrics (Strict Standards):**
{selected_rubric}

**User's Drafted Registration Text to Audit:**
\"\"\"{draft_text}\"\"\"

Determine if this text would be "Approved" or "Rejected" by an expert panel. Generate a highly detailed, professional evaluation in Traditional Chinese, and output ONLY a single JSON object.

**Required JSON Structure:**
{{
  "score": An integer between 0 and 100 representing the compliance score,
  "verdict": "建議審查通過" (if score >= 85) or "建議退件與修正" (if score < 85),
  "rejection_reasons": [
    "Specific bullet points explaining why the draft falls short of expert standards, mentioning precisely what is missing (e.g. '未提供 95% 信賴區間', '未指明排除之硬體型號'). MUST be in Traditional Chinese."
  ],
  "precise_suggestions": [
    "Surgically precise, actionable rewrite recommendations for the user. MUST be in Traditional Chinese."
  ],
  "suggested_optimized_draft": "A beautifully rewritten exemplary version of the text that integrates ALL missing elements based on the rubrics. Use brackets like [請填寫...] for missing concrete data so the user can easily edit, copy-paste, and submit. This text MUST be in Traditional Chinese and represent a highly professional, medical-grade description."
}}
"""

        # 4. 呼叫 Gemini Quality model (動態覆寫配置以避免 thread-local 認證問題)
        response = model.generate_content([prompt], generation_config={"response_mime_type": "application/json", "temperature": 0.2})
        result = json.loads(response.text)
        
        # 安全主權與在地術語過濾後處理
        if isinstance(result, dict):
            if "suggested_optimized_draft" in result:
                result["suggested_optimized_draft"] = sovereign_filter(result["suggested_optimized_draft"])
            if "rejection_reasons" in result and isinstance(result["rejection_reasons"], list):
                result["rejection_reasons"] = [sovereign_filter(r) for r in result["rejection_reasons"]]
            if "precise_suggestions" in result and isinstance(result["precise_suggestions"], list):
                result["precise_suggestions"] = [sovereign_filter(s) for s in result["precise_suggestions"]]
                
        return result

    except Exception as e:
        logging.error(f"Error during direct draft audit for item '{principle_title}': {e}", exc_info=True)
        return {
            "score": 0,
            "verdict": "審查系統異常",
            "rejection_reasons": [f"分析過程中發生異常錯誤：{str(e)}"],
            "precise_suggestions": ["請稍後再試，或聯繫系統管理員確認 API 設定。"],
            "suggested_optimized_draft": "系統暫時無法生成修改範本。"
        }
    
def _safe_float(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_metadata(metadata: dict, file_name: str = "", full_text: str = "") -> dict:
    """補齊常用補充報告欄位，降低 UI 空白比例。"""
    if not isinstance(metadata, dict):
        metadata = {}

    document_subject = build_document_search_subject(metadata, file_name, full_text)
    defaults = {
        "name_zh": document_subject,
        "name_en": "",
        "summary_zh": "文件未明確提供摘要，請由使用者確認。",
        "summary_en": "",
        "clinical_use_zh": "文件未明確載明臨床用途，請由使用者確認。",
        "target_population_zh": "文件未明確載明適用族群，請由使用者確認。",
        "input_data_zh": "文件未明確載明輸入資料，請由使用者確認。",
        "output_result_zh": "文件未明確載明輸出結果，請由使用者確認。",
        "lifecycle_plan": "文件未明確載明 AI 生命週期管理計畫，請由使用者補充。",
        "monitoring_plan": "文件未明確載明部署後監測計畫，請由使用者補充。",
        "update_plan": "文件未明確載明版本更新或再驗證計畫，請由使用者補充。",
    }
    numeric_fields = ["auc", "accuracy", "sensitivity", "specificity", "ppv", "npv"]

    for key, value in defaults.items():
        if not metadata.get(key):
            metadata[key] = value
    for key in numeric_fields:
        metadata[key] = _safe_float(metadata.get(key), 0.0)

    return metadata

def extract_metadata(full_text: str, file_name: str = "") -> dict:
    """
    從文件全文中提取補充報告資訊（中繼資料）。
    """
    try:
        st.info("正在提取補充資訊...")
        prompt = f"""
You are a Traditional Chinese data entry assistant for medical AI review reports.
Extract as much useful report information as possible from the uploaded PDF text and filename.

Rules:
- Return only one JSON object.
- Text fields MUST be in Traditional Chinese unless the field name explicitly asks for English.
- Do not leave text fields empty if the filename, title, abstract, introduction, method, or conclusion gives a reasonable clue.
- If the document does not explicitly state a text field, write a concise Traditional Chinese note such as "文件未明確載明，請由使用者確認。"
- Numeric performance fields must be numbers between 0 and 1. Use 0.0 only when no explicit numeric value is found.
- Prefer concrete wording from the document. Do not invent regulatory approvals or performance numbers.

**Required JSON Structure:**
{{
  "name_zh": "The AI model's name in Traditional Chinese.",
  "name_en": "The AI model's name in English.",
  "summary_zh": "A brief summary of the AI model in Traditional Chinese (max 50 characters).",
  "summary_en": "A brief summary of the AI model in English (max 50 characters).",
  "clinical_use_zh": "The clinical use or intended purpose in Traditional Chinese.",
  "target_population_zh": "Applicable patient group or use setting in Traditional Chinese.",
  "input_data_zh": "Input data or input modality in Traditional Chinese.",
  "output_result_zh": "Model output or result type in Traditional Chinese.",
  "auc": "The AUC value. (float, default 0.0)",
  "accuracy": "The Accuracy value. (float, default 0.0)",
  "sensitivity": "The Sensitivity value. (float, default 0.0)",
  "specificity": "The Specificity value. (float, default 0.0)",
  "ppv": "The Positive Predictive Value (PPV). (float, default 0.0)",
  "npv": "The Negative Predictive Value (NPV). (float, default 0.0)",
  "lifecycle_plan": "A summary of the AI lifecycle management plan. (string)",
  "monitoring_plan": "A summary of the post-deployment monitoring plan. (string)",
  "update_plan": "A summary of the version update and retraining plan. (string)"
}}

--- FILENAME ---
{file_name}

--- DOCUMENT TEXT (first 10000 characters) ---
{full_text[:10000]}
--- END OF TEXT ---

Now, generate only the JSON object based on the text.
"""
        # 使用高品質的全局模型進行資訊提取 (動態覆寫配置以避免 thread-local 認證問題)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json", "temperature": 0.0})
        return normalize_metadata(json.loads(response.text), file_name, full_text)
    except Exception as e:
        logging.error(f"Failed to extract metadata: {e}", exc_info=True)
        st.warning("無法自動提取補充資訊，將使用預設值。")
        return normalize_metadata({}, file_name, full_text)

def run_full_analysis(full_text, file_name="", use_multimodal=True, max_img_pages=8):
    """
    執行高度最佳化的多執行緒並行合規分析流程。
    """
    # 1. 取得歷史 RAG 資料
    rag_df = get_rag_df_from_github()

    # 2. 準備多模態資料：文字 + 圖片
    doc = fitz.open(stream=BytesIO(full_text), filetype="pdf")
    page_texts = []
    page_images = []
    for page_num, page in enumerate(doc):
        page_texts.append(page.get_text())
        # 如果啟用多模態，且在最大頁數限制內，則將頁面渲染為圖片
        if use_multimodal and page_num < max_img_pages:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0)) # 使用標準解析度節省 Token
                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))
                img.load() # 強制在主執行緒立即載入並解碼完整的圖片位元流，防範子執行緒並行讀取時 lazy loading 的 truncated image OSError 與 AssertionError 錯誤
                page_images.append(img)
            except Exception as e:
                logging.error(f"Failed to render page {page_num} as image: {e}")
                
    combined_text = "\n".join(page_texts)

    # 3. 先提取文件資訊，供補充報告使用
    with st.status("正在提取補充資訊...", expanded=False) as status:
        metadata = extract_metadata(combined_text, file_name)
        status.update(label="補充資訊提取完成", state="complete")

    all_items = TRANSPARENCY_9 + GOVERNANCE_2
    
    # 建立固定長度的列表以保持原順序
    results_t = [None] * 9
    results_g = [None] * 2

    # 使用 Streamlit 狀態元件並行更新進度
    with st.status("🚀 正在並行分析合規指標...", expanded=True) as status_block:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # 內嵌包裝函式
        def analyze_item_wrapper(idx, item):
            try:
                # 執行單項分析
                res = perform_direct_analysis(item, combined_text, page_images, rag_df)
                return idx, res
            except Exception as e:
                logging.error(f"Error in parallel analysis of item {item['title']}: {e}")
                return idx, normalize_analysis_result({"status": "Analysis Error", "summary": f"分析失敗: {e}"})

        completed_count = 0
        total_items = len(all_items)
        
        # 使用 ThreadPoolExecutor 並行分析 11 項合規指標
        # 5 個 Worker 是兼顧 API rate limits 與並行效能的最佳配置
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(analyze_item_wrapper, idx, item): (idx, item)
                for idx, item in enumerate(all_items)
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx, item = futures[future]
                try:
                    index, res = future.result()
                    if index < 9:
                        results_t[index] = res
                    else:
                        results_g[index - 9] = res
                except Exception as exc:
                    logging.error(f"Item {item['title']} generated an exception: {exc}")
                    err_res = normalize_analysis_result({"status": "Analysis Error", "summary": f"發生異常: {exc}"})
                    if idx < 9:
                        results_t[idx] = err_res
                    else:
                        results_g[idx - 9] = err_res
                
                completed_count += 1
                progress_bar.progress(completed_count / total_items)
                status_text.write(f"✅ 已完成分析: **{item['title']}** ({completed_count}/{total_items})")
                
        status_block.update(label="🎉 所有指標分析完成！", state="complete")
        progress_bar.empty()
        status_text.empty()
            
    return {"t": results_t, "g": results_g, "metadata": metadata}

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
            "狀態": _translate_status_to_zh(item.get('status', 'Unknown')),
            "符合機率": f"{item.get('pass_probability', 0)}%",
            "摘要": item.get('summary_zh', '無摘要或翻譯失敗'),
            "建議": item.get('suggestion_zh', '')
        })
    # 處理 2 大指標
    for i, item in enumerate(st.session_state['res_g']):
        data.append({
            "分類": "核心治理指標",
            "項目": GOVERNANCE_2[i]['title'],
            "狀態": _translate_status_to_zh(item.get('status', 'Unknown')),
            "符合機率": f"{item.get('pass_probability', 0)}%",
            "摘要": item.get('summary_zh', '無摘要或翻譯失敗'),
            "建議": item.get('suggestion_zh', '')
        })
    
    df = pd.DataFrame(data)
    # 使用 StringIO 轉為 CSV 字串
    return df.to_csv(index=False).encode('utf-8-sig')

# ---------- 4. UI 介面 ----------

def generate_radar_chart(results_t):
    """根據九大透明性指標的符合機率繪製雷達圖"""
    if not results_t:
        return None
        
    categories = [item["title"] for item in TRANSPARENCY_9]
    # 取得符合機率
    values = [res.get('pass_probability', 0) for res in results_t]
    
    # 為了使雷達圖閉合，需要將第一個點重複加到最後
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.15)',  # 品牌紫羅蘭透光填滿
        line=dict(color='rgb(99, 102, 241)', width=2.5),
        name='符合機率 (%)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%',
                gridcolor='rgba(156, 163, 175, 0.25)',
                linecolor='rgba(156, 163, 175, 0.25)',
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                gridcolor='rgba(156, 163, 175, 0.25)',
                linecolor='rgba(156, 163, 175, 0.25)',
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        margin=dict(l=45, r=45, t=30, b=30),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def format_percentage_metric(val):
    """將比例格式化為百分比字串，兼顧大於 1 已經乘以 100 的數字。"""
    val_f = _safe_float(val)
    if val_f == 0.0:
        return "0.0%"
    if val_f > 1.0:
        return f"{val_f:.1f}%"
    return f"{val_f * 100:.1f}%"

def main():
    st.set_page_config(page_title="醫療 AI 治理檢核", layout="wide")
    st.title("🛡️ 負責任 AI 自動檢核系統")

    if 'res_t' not in st.session_state:
        st.session_state['res_t'] = None
    if 'metadata' not in st.session_state:
        st.session_state['metadata'] = {}
    if 'rag_selection' not in st.session_state:
        st.session_state.rag_selection = None
    if 'rag_old_feedback_content' not in st.session_state:
        st.session_state.rag_old_feedback_content = ""
    if 'rag_new_principle_content' not in st.session_state:
        st.session_state.rag_new_principle_content = ""

    # 在 UI 中顯示當前模式
    mode_display = "☁️ 雲端智慧分析模式"
    st.subheader(mode_display)

    with st.sidebar:
        st.title(" ") # 佔位符
        st.divider()
        st.caption("本網站內容由人工智慧生成，僅為參考用途。")
        st.caption("聯絡信箱：AnsonHsieh@itri.org.tw")

    # 建立雙重檢核模式的分頁
    tab_pdf_mode, tab_draft_mode = st.tabs([
        "📑 計畫書全文 PDF 檢核 (Full PDF Analysis)", 
        "✍️ 登錄內文專家檢核 (Direct Draft Audit)"
    ])

    with tab_pdf_mode:
        # --- 步驟 1: 上傳檔案 ---
        st.header("1. 上傳計畫書")
        pdf_file = st.file_uploader("上傳您的醫療器材軟體計畫書 (PDF)", type="pdf", help="系統將自動分析文件內容，評估其是否符合相關治理原則。", label_visibility="collapsed")

        # --- 步驟 1.5: 分析設定 ---
        st.markdown("##### ⚙️ 分析引擎設定")
        col_toggle1, col_toggle2 = st.columns(2)
        with col_toggle1:
            use_multimodal = st.toggle("📷 啟用多模態圖片分析", value=True, help="開啟後，除了分析 PDF 的文字，也會傳送頁面截圖給自動化審查引擎進行視覺分析（適合含有流程圖、統計圖表或介面截圖的計畫書）。若遇到 Token 限制或分析速度較慢時，可嘗試關閉。")
        with col_toggle2:
            max_img_pages = st.slider("🖼️ 限制多模態分析最大頁數", min_value=1, max_value=20, value=8, help="為節省 Token 並加速分析，僅會將 PDF 前 N 頁渲染為圖片傳送，其餘頁面將使用文字進行分析。")

        # --- 步驟 2: 執行分析 ---
        if pdf_file:
            st.header("2. 執行分析")
            if st.button("🚀 開始分析", use_container_width=True):
                with st.spinner("正在讀取檔案並使用雲端模型進行並行分析..."):
                    # 步驟 1: 執行核心並行分析 (模型已在 analyze 中直接輸出繁中成果，免除二次翻譯)
                    results = run_full_analysis(pdf_file.getvalue(), pdf_file.name, use_multimodal=use_multimodal, max_img_pages=max_img_pages)
                    
                    st.session_state['res_t'] = results['t']
                    st.session_state['res_g'] = results['g']
                    st.session_state['metadata'] = results.get('metadata', {})
                    
                    # 重設 RAG 表單狀態以反映新 analysis
                    st.session_state.rag_selection = None
                    st.session_state.rag_old_feedback_content = ""
                    st.session_state.rag_new_principle_content = ""

                    # 步驟 3: 自動保存本次分析的完整結果
                    with st.status("正在歸檔分析紀錄...", expanded=False) as status:
                        if save_analysis_history_to_github(results, pdf_file.name):
                            status.update(label="分析紀錄歸檔成功", state="complete")
                        else:
                            status.update(label="分析紀錄歸檔失敗", state="error")
                    
                    # 步驟 4: 上傳日誌檔案
                    with st.status("正在上傳日誌檔案...", expanded=False) as status:
                        if upload_log_to_github():
                            status.update(label="日誌檔案上傳成功", state="complete")
                        else:
                            status.update(label="日誌檔案上傳失敗", state="error")

        # 顯示結果
        if st.session_state.get('res_t'):
            st.header("3. 檢視分析結果")
            
            # 顯示雷達圖與關鍵量化表現指標
            col_radar, col_metrics = st.columns([11, 9])
            with col_radar:
                st.subheader("🕸️ 治理合規雷達圖 (Compliance Radar)")
                radar_fig = generate_radar_chart(st.session_state['res_t'])
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.info("尚無資料繪製雷達圖")
                    
            with col_metrics:
                st.subheader("📈 關鍵量化表現指標")
                metadata = st.session_state.get('metadata', {})
                
                # 卡片排版顯示從 metadata 中提取出的成效指標
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("AUC 面積", f"{_safe_float(metadata.get('auc', 0.0)):.2f}")
                    st.metric("靈敏度 (Sensitivity)", format_percentage_metric(metadata.get('sensitivity', 0.0)))
                    st.metric("陽性預測值 (PPV)", format_percentage_metric(metadata.get('ppv', 0.0)))
                with m_col2:
                    st.metric("準確度 (Accuracy)", format_percentage_metric(metadata.get('accuracy', 0.0)))
                    st.metric("特異度 (Specificity)", format_percentage_metric(metadata.get('specificity', 0.0)))
                    st.metric("陰性預測值 (NPV)", format_percentage_metric(metadata.get('npv', 0.0)))

            st.divider()
            st.subheader("📊 九大透明性指標細節")
            t_data = st.session_state['res_t']
            for r in range(3):
                cols = st.columns(3)
                for c in range(3):
                    idx = r * 3 + c
                    if idx < len(t_data):
                        item = t_data[idx]
                        with cols[c]:
                            # 顯示標題和分數
                            prob = item.get('pass_probability', 0)
                            st.subheader(f"{idx+1}. {TRANSPARENCY_9[idx]['title']}")
                            st.markdown(f"**符合機率:** {prob}%")
                            st.divider()
                            
                            # 使用 .get() 安全地讀取所有可能缺失的鍵，並提供預設值
                            status_en = item.get('status', 'Unknown')
                            color = "green" if status_en == "Exists" else "red"
                            status_zh = _translate_status_to_zh(status_en)
                            source_text = item.get('source', '未知')
                            summary_text_zh = item.get('summary_zh', '無摘要或翻譯失敗')
                            suggestion_text_zh = item.get('suggestion_zh', '')

                            st.markdown(f"**狀態:** :{color}[{status_zh}] | **來源:** {source_text}")
                            st.info(f"**摘要:** {summary_text_zh}")
                            if suggestion_text_zh: # 只有當 suggestion 存在且非空時才顯示
                                st.warning(f"💡 建議與修補指引：{suggestion_text_zh}")

            st.divider()
            st.subheader("📋 核心治理指標")
            g_data = st.session_state['res_g']
            df_g = pd.DataFrame([{
                "評估項目": GOVERNANCE_2[i]['title'],
                "狀態": _translate_status_to_zh(d.get('status', 'Unknown')),
                "符合機率": f"{d.get('pass_probability', 0)}%",
                "摘要": d.get('summary_zh', '無摘要或翻譯失敗'),
                "建議與指引": d.get('suggestion_zh', ''),
            } for i, d in enumerate(g_data)])
            st.table(df_g)

            # ---------- 下載報告區塊 ----------
            st.header("4. 後續操作")
            tab1, tab2, tab3 = st.tabs(["📥 匯出報告", "📝 學習與優化 (歷史知識庫)", "📄 補充報告資訊 (選填)"])

            with tab1: # 匯出報告
                csv_data = convert_results_to_csv()
                if csv_data:
                    st.download_button(
                        label="💾 下載 CSV 報告",
                        data=csv_data,
                        file_name=f"醫療AI檢核報告_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption("按鈕將下載包含透明性原則與治理指標的完整彙總表格。")

            with tab2: # RAG 回饋
                st.caption("您的回饋建議將存入 AI 知識庫，用於強化未來分析結果。")
                all_titles = [i['title'] for i in (TRANSPARENCY_9 + GOVERNANCE_2)]

                # 如果選擇為空（例如，在新分析之後），則將其預設為列表中的第一項。
                if st.session_state.rag_selection is None and all_titles:
                    st.session_state.rag_selection = all_titles[0]

                # 如果舊回饋的內容為空，請根據當前選擇進行填充。
                # 這確保了在第一次載入時內容會被正確設置。
                if not st.session_state.rag_old_feedback_content and all_titles:
                    result_lookup = build_result_lookup(st.session_state.get('res_t'), st.session_state.get('res_g'))
                    initial_result = result_lookup.get(st.session_state.rag_selection, {})
                    st.session_state.rag_old_feedback_content = format_analysis_result_for_feedback(st.session_state.rag_selection, initial_result)

                # selectbox 現在使用一個 key 和一個 on_change 回呼函式來管理狀態，而不是在每次執行時重新計算。
                st.selectbox(
                    "選擇要修正的項目",
                    all_titles,
                    key='rag_selection',
                    on_change=update_rag_feedback_text
                )

                with st.form("rag_feedback_form"):
                    # 文字區域現在透過其 key 綁定到 session_state。
                    st.text_area(
                        "使用者原始回饋 (Old)",
                        value=st.session_state.rag_old_feedback_content,
                        height=180,
                        key="rag_old_feedback_content",
                        help="預設帶入此項目的 AI 分析結果，可直接修改成你認為正確的問題描述。"
                    )
                    st.text_area("泛化後的新原則 (New)", value=st.session_state.rag_new_principle_content,
                                 placeholder="例如：「應交叉比對內文、圖表和表格中的效能指標數據，確保其一致性。」", key="rag_new_principle_content")
                    submit_rag = st.form_submit_button("✅ 送出經驗並優化未來分析")

                    if submit_rag:
                        if not GITHUB_TOKEN: st.error("請檢查 GITHUB_TOKEN 設定。")
                        elif not st.session_state.rag_new_principle_content: st.warning("請填寫「泛化後的新原則」。")
                        else:
                            with st.spinner("同步至 GitHub 中..."):
                                feedback_to_save = f"原始分析結果：\n{st.session_state.rag_old_feedback_content}\n\n修正後通用原則：\n{st.session_state.rag_new_principle_content}"
                                if update_rag_to_github(st.session_state.rag_selection, feedback_to_save):
                                    st.success("回饋成功！下次分析將參考此經驗。")
                                else:
                                    st.error("寫入失敗，請確認 Token 權限。")
            
            with tab3: # 手動填寫資訊
                st.info("您可以在此手動填寫額外資訊，這些資訊將會被整合到未來的報告匯出功能中。")
                metadata = st.session_state.get('metadata', {})
                with st.expander("模型基本資料", expanded=True):
                    st.text_input("AI模型名稱", value=metadata.get('name_zh', ''), key="manual_name_zh")
                    st.text_input("AI Model Name", value=metadata.get('name_en', ''), key="manual_name_en")
                    st.text_area("AI模型摘要", value=metadata.get('summary_zh', ''), max_chars=50, help="摘要上限 50 字", key="manual_summary_zh")
                    st.text_area("AI Model Summary", value=metadata.get('summary_en', ''), max_chars=50, help="Summary max 50 characters", key="manual_summary_en")
                with st.expander("臨床使用資訊", expanded=True):
                    st.text_area("臨床用途", value=metadata.get('clinical_use_zh', ''), key="manual_clinical_use")
                    st.text_area("適用族群或使用場域", value=metadata.get('target_population_zh', ''), key="manual_target_population")
                    st.text_area("輸入資料", value=metadata.get('input_data_zh', ''), key="manual_input_data")
                    st.text_area("輸出結果", value=metadata.get('output_result_zh', ''), key="manual_output_result")
                with st.expander("成效指標 (Performance Metrics)", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.number_input("AUC", value=_safe_float(metadata.get('auc', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_auc")
                        st.number_input("陽性預測值 (PPV)", value=_safe_float(metadata.get('ppv', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_ppv")
                    with col2:
                        st.number_input("準確度 (Accuracy)", value=_safe_float(metadata.get('accuracy', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_acc")
                        st.number_input("陰性預測值 (NPV)", value=_safe_float(metadata.get('npv', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_npv")
                    with col3:
                        st.number_input("靈敏度 (Sensitivity)", value=_safe_float(metadata.get('sensitivity', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_sens")
                        st.number_input("特異度 (Specificity)", value=_safe_float(metadata.get('specificity', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="manual_spec")
                with st.expander("維運計畫 (Maintenance Plan)", expanded=True):
                    st.text_area("AI生命週期管理", value=metadata.get('lifecycle_plan', ''), help="AI Lifecycle Management", key="manual_lifecycle")
                    st.text_area("監測指標計畫", value=metadata.get('monitoring_plan', ''), help="Monitoring Plan", key="manual_monitoring")
                    st.text_area("版本更新計畫", value=metadata.get('update_plan', ''), help="Version Update Plan", key="manual_update")

    with tab_draft_mode:
        st.header("✍️ 登錄內文專家審查與檢核 (Expert Draft Review)")
        st.markdown("如果您正在為臨床 AI 登錄平台撰寫填報內容，可以直接在此針對特定原則進行**合規預檢與專家級輔助修正**，減少被審查專家退件的機率。")
        
        all_items = TRANSPARENCY_9 + GOVERNANCE_2
        item_titles = [item['title'] for item in all_items]
        
        # Selectbox for principle
        selected_title = st.selectbox(
            "選擇您要檢核的登錄指標項目",
            item_titles,
            key="draft_principle_select",
            help="選擇目前正在撰寫的指標，系統將自動載入專家系統硬性審核標準 (Rubrics)。"
        )
        
        # Find selected item
        selected_item = next(i for i in all_items if i['title'] == selected_title)
        
        # Show description of principle
        st.info(f"💡 **該項指標官方標準定義：**\n{selected_item['desc']}")
        
        # Text area for user draft
        draft_text = st.text_area(
            "請在此輸入您擬定填報的說明草稿 (Draft Text)",
            height=220,
            key="draft_text_input",
            placeholder="請輸入您為本項指標撰寫之計畫書內文或申報說明（例如訓練集人口分布、AUC、95% CI 或是生命週期管理規劃等...）"
        )
        
        # Button to run draft audit
        if st.button("🚀 執行專家系統審核 (Run Audit)", use_container_width=True, key="btn_draft_audit"):
            if not draft_text.strip():
                st.warning("請先輸入您的填報草稿內容。")
            else:
                with st.spinner("正在對照審查知識庫並比對臨床審查標準進行專家評估..."):
                    # Load RAG
                    rag_df = get_rag_df_from_github()
                    # Audit
                    audit_res = audit_direct_draft(selected_title, selected_item['desc'], draft_text, rag_df)
                    
                    # Store results in session state
                    st.session_state['draft_audit_results'] = audit_res
                    st.session_state['draft_audit_principle'] = selected_title
                    
        # Render draft audit results if exists
        if st.session_state.get('draft_audit_results') and st.session_state.get('draft_audit_principle') == selected_title:
            res = st.session_state['draft_audit_results']
            score = res.get('score', 0)
            verdict = res.get('verdict', '建議退件與修正')
            
            st.divider()
            st.subheader("📊 專家系統審查評估報告")
            
            # Show score and verdict
            score_col, verdict_col = st.columns(2)
            with score_col:
                st.metric("專家系統合規評分 (Compliance Score)", f"{score}%")
            with verdict_col:
                verdict_color = "green" if verdict == "建議審查通過" else "red"
                st.markdown(f"### 專家審查意見：:{verdict_color}[{verdict}]")
                
            st.divider()
            
            # Show rejection reasons and suggestions
            reasons_col, suggestions_col = st.columns(2)
            with reasons_col:
                st.subheader("❌ 專家判定缺失 / 退件原因")
                reasons = res.get('rejection_reasons', [])
                if reasons:
                    for r in reasons:
                        st.markdown(f"- {r}")
                else:
                    st.success("🎉 無顯著缺失！草稿結構極其完整，符合第二階段專家審核標準。")
                    
            with suggestions_col:
                st.subheader("💡 專家修改與補充指引")
                suggestions = res.get('precise_suggestions', [])
                if suggestions:
                    for s in suggestions:
                        st.markdown(f"- {s}")
                else:
                    st.info("符合最高專家標準，無須額外修改。")
                    
            st.divider()
            
            # Show suggested optimized draft
            st.subheader("📝 專家級修正後推薦填報範本")
            st.markdown("我們已為您將缺失內容補齊並重寫為極具專業感之學術範本。您可以直接點擊右上角複製，修改括號 `[...]` 內之數值即可直接填入登錄平台！")
            
            optimized_draft = res.get('suggested_optimized_draft', '')
            st.text_area(
                "推薦修改範本 (可直接複製)",
                value=optimized_draft,
                height=250,
                key="suggested_optimized_draft_box"
            )

if __name__ == "__main__":
    main()
