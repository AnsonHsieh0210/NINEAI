負責任 AI 自動檢核系統 (Medical AI Audit System) — v3.1 專家 RAG 強化版
本專案是一套專為醫療 AI 專案管理與 IRB / TFDA 計畫書審查打造的自動化合規檢核平台。系統採用 Streamlit 建立前端互動介面，並以 Gemini 3.1 Pro 語言模型為核心推理引擎，協助研究人員與審查委員快速、深度地評估計畫書的合規性。

🚀 專案核心價值與技術架構
為了建立一個「既有專業法規標準，又具備臨床審查深度」的 AI，本系統打破了傳統單純改 Prompt 的瓶頸，採用了雙層知識隔離架構 (Hybrid Knowledge Architecture)：

介面層（官方標準定義）：
前端 UI 的九大透明性原則與治理指標完全採用官方標準、簡潔的定義。解決了過去直接修改欄位描述導致介面流於特定專案口吻、偏離標準法規的疑慮，維持系統的行政合規性與專業觀感。

邏輯層（專家背景知識 RAG）：
將臨床醫師與審查專家的「硬性門檻」與「判定 Rubrics」內化於後台的 RAG (檢索增強生成) 提示詞與向量比對中。AI 在後台會以極其嚴格的專家標準（如：量化指標必須有 95% CI、外部驗證必須是跨中心數據等）進行實質審查。

此外，專案具備 「數據閉環 (Data Feedback Loop)」 能力，醫師在介面上送出的每筆修正回饋，都會自動透過 GitHub API 同步回儲存庫的 RAG.csv，使系統具備持續演進、內化「醫師靈魂」並轉向 AI Studio Fine-tuning (模型微調) 的長期發展潛力。

📊 檢核指標說明
系統自動針對醫療計畫書進行多維度的 9+2 關鍵指標 深度合規檢核：

九大透明性原則 (Transparency 9)
介入詳情及輸出：驗證模型輸出形式（如風險評分、量化區間）與臨床解讀指引。

介入目的：評估臨床用途（輔助診斷、分流、篩查）及目標疾病、受測族群之限制。

警告與範圍外使用：檢查設備限制與明確的排除準則 (Exclusion Criteria)。

開發詳情及輸入特徵：審核訓練資料的時空分佈、特徵維度與演算法架構。

確保公平性的過程：檢查是否針對敏感屬性（性別、年齡、種族）進行性能分層分析。

外部驗證過程：從嚴審查是否具備完全獨立的中心驗證數據，或跨設備硬體（如 GE, Philips）之相容性測試。

量化表現指標：檢查統計數據是否完整（如包含 95% 信賴區間、NPV/PPV 等臨床指標）。

持續維護與監控：評估部署後的模型漂移 (Model Drift) 監測指標與錯誤處理機制。

更新與持續驗證計畫：審查模型再訓練 (Retraining) 的頻率與更新後的效能驗證流程。

核心治理指標 (Governance 2)
可解釋性分析：評估技術解釋（如 Heatmap）是否與臨床放射徵象對照，以及醫師如何驗證模型結果。

AI生命週期管理：審查從開發、上市後監測 (Post-market Surveillance) 到退場機制的全流程風險評估。

🛠️ 安裝與本地啟動指南
1. 複製專案與環境準備
請確保您的系統已安裝 Python 3.9+，並複製本專案至本地：

Bash
git clone https://github.com/AnsonHsieh0210/NINEAI.git
cd NINEAI
2. 安裝必要套件
首先，建立一個 `requirements.txt` 檔案，其中包含所有必要的套件。接著，使用 pip 一次性安裝所有依賴：

Bash
pip install -r requirements.txt
3. 配置環境變數 (.env)
在專案根目錄下建立一個 .env 檔案，並配置您的 Google AI API Key 與 GitHub Personal Access Token：

Plaintext
GOOGLE_API_KEY=your_gemini_api_key_here
GITHUB_TOKEN=your_github_token_here
💡 備註：GITHUB_TOKEN 需具備讀寫該倉儲內 RAG.csv 的權限，以便系統能自動寫入醫師的臨床回饋經驗。

4. 啟動 Streamlit 服務
Bash
streamlit run app.py
啟動後，瀏覽器將自動開啟本地網頁（預設為 http://localhost:8501）。

📈 專案未來演進：走向 AI Studio 微調
當本專案的 RAG.csv 透過網頁端的回饋表單累積了足夠的醫師修正資料（建議 30–50 筆以上）後，即可啟動模型內化閉環：

格式轉換：使用轉換腳本將 RAG.csv 導出為符合 Google AI Studio 格式的 jsonl 對話訓練集。

AI Studio 微調：在 Google AI Studio 中建立一個以 gemini-3.1-pro 為基底的 Tuned Model，上傳該資料集，並設定 Hyperparameters（建議 Epochs 設為 3-5 輪以避免過擬合）。

無縫接軌：微調完成後，僅需將程式碼中的 CURRENT_MODEL_ID 變數修改為您專屬的微調模型 ID（例如：tunedModels/medical-expert-v31-xxxx），系統即可在不依賴冗長 Prompt 的情況下，直覺且深層地吐出具備「醫師魂」的專業審查報告。