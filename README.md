# 🌟 ZenitAi — AI English Teacher Pro
==========================

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA%203.3--70B-orange)](https://console.groq.com)
[![spaCy](https://img.shields.io/badge/NLP-spaCy%203.7-green)](https://spacy.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Professional AI-Powered English Language Platform**  
> Groq LLaMA 3.3-70B · spaCy NLP · 40+ Features · Dark Professional Theme

---

## 📚 Table of Contents
1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [Installation](#-installation)
4. [Deployment (Streamlit Cloud)](#-deployment-streamlit-cloud)
5. [Project Structure](#-project-structure)
6. [API Key Setup](#-api-key-setup)
7. [Contributing](#-contributing)
8. [License](#-license)

---

## 🎯 Features

### Grammar & Writing
- 🛡️ **Grammar Checker** — AI-powered issue detection & one-click fixes
- ✍️ **Style Transformer** — 4 styles × 4 tones (Academic / Casual / Business / Creative)
- 💬 **Smart Paraphraser** — 3 formality levels (Formal / Neutral / Informal)
- 🎓 **AI Grammar Fix** — Groq LLaMA rewrites your text with explanations

### Linguistic Analysis
- ⏳ **12-Tense Matrix** — Convert any sentence into all 12 English tenses instantly
- 🎨 **POS Tag Painter** — Color-coded Part-of-Speech visualization
- 📊 **Word Frequency Lab** — Bar charts, donut charts, stopword filtering
- 🔭 **Deep NLP Analysis** — Sentiment, Named Entities, Dependency Parsing

### Translation (AI-Powered)
- 🌐 English ↔ Urdu
- 🌐 English ↔ Pashto
- 🌐 English ↔ Hindi
- 🌐 English ↔ Arabic

### AI Tutor & Learning
- 💡 **AI Tutor** — 4 modes: Explain / Teach / Simplify / Breakdown
- 📚 **Practice Mode** — MCQ Quiz Generator with instant scoring
- 📖 **Vocabulary Builder** — Daily words, synonyms & antonyms

### Analytics & Tracking
- 📈 **Session Analytics** — Track analyses, AI calls, and score
- 📊 **Progress Dashboard** — Real-time metrics in the sidebar

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Streamlit |
| AI / LLM | Groq LLaMA 3.3-70B |
| NLP | spaCy `en_core_web_sm` |
| Sentiment | TextBlob |
| Language Detection | langdetect |
| Visualization | Plotly |
| Data | Pandas |

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/zenitai.git
cd zenitai

# 2. Install dependencies
pip install -r requirements.txt


# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🚀 Deployment (Streamlit Cloud)

### Step 1 — Push to GitHub
```bash
git add app.py requirements.txt README.md
git commit -m "Initial ZenitAi deployment"
git push origin main
```

### Step 2 — Connect on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → select your repository
3. Set **Main file path** to `app.py`
4. Click **Deploy**

### Step 3 — Add Your API Key as a Secret
In your Streamlit Cloud dashboard → **Settings → Secrets**, add:

```toml
[groq]
api_key = "gsk_your_groq_api_key_here"
```

> 🔑 **Get your free Groq API key** at [console.groq.com](https://console.groq.com)

No key in the code — no security risk on GitHub. ✅

---

## 🗂️ Project Structure

```
zenitai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── secrets.toml        # Local API key (DO NOT commit this file)
```


## 🎨 Design

- **Theme**: Verdante Teal — Professional Dark Mode
- **Primary**: `#4DD9F0` (Electric Teal)
- **Background**: `#06181C` → `#08262C` (Deep Ocean)
- **Fonts**: DM Serif Display (headings) · DM Sans (body)
- **Layout**: Wide, fully responsive

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure your code follows existing style conventions and all features remain intact.

---

## 📜 License

ZenitAi is licensed under the [MIT License](LICENSE).  
By contributing, you agree to abide by its terms.

---

<p align="center">
  🌟 <b>ZenitAi Pro v1.0</b> — Powered by Groq LLaMA 3.3 + spaCy NLP<br/>
  Built with ❤️ for English learners worldwide
</p>
