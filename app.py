"""
------------------------------------------------------------------------
AI Real Estate Agent - Case Study Implementation
------------------------------------------------------------------------
Author:     Harsh Verma
Date:       November 2025
Context:    Technical Submission for Agent Mira

License:    This code is provided solely for evaluation purposes.
            It remains the intellectual property of the author.
------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import re
import requests
import json
import io
import base64
import matplotlib # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# --- Configuration ---
REAL_ESTATE_TOPICS = [
    "backyard", "pool", "garage", "view", "quiet", "school", "modern", 
    "renovated", "kitchen", "fireplace", "gym", "garden", "spacious", 
    "natural light", "subway", "transit", "park", "safe", "luxury", "open concept"
]

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

# --- Pydantic Model ---
class UserQuery(BaseModel):
    budget: str
    bedrooms: int
    bathrooms: int
    description: str

# --- Core Logic ---
class RealEstateAgent:
    def __init__(self):
        self.model = None
        self.props = None
        self.prop_embeddings = None
        self.index = None

    def load_resources(self, props_path):
        print("1. Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("2. Loading Property Data...")
        self.props = pd.read_excel(props_path, sheet_name='Property Data')
        self.props["Price_num"] = self.props["Price"].apply(self._parse_money)
        self.props["Qualitative Description"] = self.props["Qualitative Description"].fillna("").astype(str)
        
        print("3. Creating Vector Index (FAISS)...")
        embeddings = self.model.encode(
            self.props["Qualitative Description"].tolist(), 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        self.prop_embeddings = np.array(embeddings).astype('float32')
        
        # IndexFlatIP = Cosine Similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.prop_embeddings.shape[1])
        self.index.add(self.prop_embeddings)
        print("--- System Ready ---")

    def _parse_money(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        s = str(val).strip().lower().replace("$", "").replace(",", "")
        if s.endswith("k"): return float(s[:-1]) * 1000.0
        if s.endswith("m"): return float(s[:-1]) * 1_000_000.0
        try: return float(s)
        except: return 0.0

    def _extract_topics(self, text):
        text = text.lower()
        return [t for t in REAL_ESTATE_TOPICS if re.search(r'\b' + t + r'\b', text)]

    def _calculate_hard_score(self, user_budget, user_beds, prop_row):
        score = 100.0
        log = []
        
        # --- 1. Price Logic ---
        price_diff = prop_row["Price_num"] - user_budget
        
        if price_diff > 0:
            chunks = price_diff / 100000.0
            penalty = chunks * 10.0 
            score -= penalty
            log.append(f"Over budget by ${price_diff:,.0f} (Penalty: -{penalty:.1f})")
        else:
            savings = user_budget - prop_row["Price_num"]
            log.append(f"Under budget by ${savings:,.0f} (Bonus)")
            
        # --- 2. Bedroom Logic ---
        bed_diff = prop_row["Bedrooms"] - user_beds
        if bed_diff < 0:
            score -= 30.0
            log.append(f"Missing {abs(bed_diff)} bedroom(s) (Penalty: -30)")
        elif bed_diff > 0:
            score += 5.0
            log.append(f"Has {int(bed_diff)} extra bedroom(s) (Bonus)")
        
        final_score = max(0.0, min(100.0, score))
        return final_score, log

    def _generate_plots(self, candidates_df):
        """Generates Base64 encoded plots for the frontend."""
        plots = {}
        
        # 1. Bar Chart: Final System Scores
        plt.figure(figsize=(8, 4))
        # FIXED: Added hue and legend=False to fix Seaborn warning
        sns.barplot(x="Property ID", y="System Score", data=candidates_df, palette="viridis", hue="Property ID", legend=False)
        plt.title("Top 5 Properties by System Match Score")
        plt.ylim(0, 100)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['bar_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # 2. Heatmap: Score Components (Explainability)
        # FIXED: These columns now exist in the dataframe
        heatmap_data = candidates_df[["Property ID", "Context Score", "Hard Score", "Topic Boost"]].set_index("Property ID")
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", vmin=0, vmax=100)
        plt.title("Score Breakdown: Why these properties matched?")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plots

    def _query_llama(self, prompt):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json"
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            return json.dumps({"error": str(e)})

    def recommend_new_user(self, query: UserQuery):
        budget_num = self._parse_money(query.budget)
        user_emb = self.model.encode([query.description], normalize_embeddings=True)[0]
        user_emb_faiss = np.array([user_emb]).astype('float32')
        user_topics = self._extract_topics(query.description)
        
        # 1. Retrieval
        k = 50 
        distances, indices = self.index.search(user_emb_faiss, k)
        valid_indices = [i for i in indices[0] if i != -1]
        candidate_props = self.props.iloc[valid_indices].copy()
        
        final_candidates = []
        
        # 2. Re-Ranking
        for idx, (row_idx, prop_row) in enumerate(candidate_props.iterrows()):
            context_sim = distances[0][idx]
            context_score = round(context_sim * 100, 1)
            
            hard_score, hard_log = self._calculate_hard_score(budget_num, query.bedrooms, prop_row)
            
            prop_topics = self._extract_topics(prop_row["Qualitative Description"])
            common_topics = set(user_topics).intersection(prop_topics)
            topic_val = len(common_topics) * 5
            
            system_score = round((context_score * 0.6) + (hard_score * 0.2) + (topic_val * 0.2), 1)
            
            final_candidates.append({
                "Property ID": int(prop_row["Property ID"]),
                "Price": prop_row["Price"],
                "Bedrooms": int(prop_row["Bedrooms"]),
                # FIXED: Renamed keys to match the Heatmap expectation
                "Context Score": context_score,   
                "Hard Score": hard_score,
                "Topic Boost": topic_val,
                "System Score": round(system_score, 1),
                "Topics": list(common_topics),
                "Analysis_Log": hard_log,
                "Description": prop_row["Qualitative Description"][:200] + "..."
            })

        df_candidates = pd.DataFrame(final_candidates).sort_values("System Score", ascending=False).head(5)
        
        # 3. Visualization
        charts = self._generate_plots(df_candidates)
        
        # 4. Agent Decision
        prompt = self._construct_llm_prompt(query, user_topics, df_candidates)
        llm_response = self._query_llama(prompt)
        
        return llm_response, charts

    def _construct_llm_prompt(self, query, user_topics, candidates_df):
        prompt = f"""
You are a strategic Real Estate Advisor. Return ONLY a JSON object.

User Context:
- Budget: {query.budget}
- Needs: {query.bedrooms} Beds
- Wants: "{query.description}"

Candidate Analysis (Data for your reasoning):
"""
        for _, row in candidates_df.iterrows():
            # FIXED: Updated row keys to match the new dictionary structure
            prompt += f"""
[ID: {row['Property ID']}]
- Price: {row['Price']}
- Specs: {row['Bedrooms']} Beds
- Vibe Match: {row['Context Score']}% 
- Key Factors: {row['Analysis_Log']}
- Desc: "{row['Description']}"
"""
        prompt += """
Your Task:
1. Select the top 3 properties.
2. WRITE "UPSELL" JUSTIFICATIONS:
   - TRUTH PROTOCOL: Check 'Key Factors'. If it says "Over budget", you MUST state exactly how much. Never say it is within budget if the log says otherwise.
   - THE UPSELL: If it is over budget but has Extra Beds or High Vibe (>90%), frame it as a trade-off.
   - If it is under budget, highlight the savings.
   - If it is not matching well, say "we will keep in mind your requirements".
3. Score each property from 0-100 based on overall fit.

Output Format (JSON ONLY):
{
  "recommendations": [
    {"Property ID": id, "Final Score": "0-100", "Justification": "Strictly follow Truth Protocol & Upsell logic above."}
  ]
}
"""
        return prompt

# --- FastAPI App ---
agent = RealEstateAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    agent.load_resources('data/Case Study 2 Data (1).xlsx')
    yield

app = FastAPI(lifespan=lifespan)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Real Estate Agent</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: auto; padding: 20px; background: #f0f2f5; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .input-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #34495e; }
        input, textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 8px; box-sizing: border-box; font-size: 14px; }
        button { background: #3498db; color: white; padding: 15px; border: none; width: 100%; cursor: pointer; font-size: 16px; border-radius: 8px; font-weight: bold; transition: background 0.3s; }
        button:hover { background: #2980b9; }
        
        .grid-results { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .result-col { display: flex; flex-direction: column; gap: 15px; }
        
        .card { background: #fff; padding: 20px; border-left: 5px solid #3498db; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .card h3 { margin: 0 0 10px 0; color: #2c3e50; }
        .score { color: #27ae60; font-weight: bold; font-size: 1.1em; }
        
        .chart-container { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; }
        .chart-container img { max-width: 100%; height: auto; border-radius: 4px; }
        
        #loading { display: none; text-align: center; color: #7f8c8d; margin-top: 20px; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏡 AI Property Matchmaker</h1>
        
        <div class="input-group">
            <label>Budget</label>
            <input type="text" id="budget" value="$850k">
        </div>
        
        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div style="flex: 1;">
                <label>Bedrooms</label>
                <input type="number" id="bedrooms" value="3">
            </div>
            <div style="flex: 1;">
                <label>Bathrooms</label>
                <input type="number" id="bathrooms" value="2">
            </div>
        </div>

        <div class="input-group">
            <label>Preferences (Vibe, Location, Amenities)</label>
            <textarea id="description" rows="3" placeholder="Modern home with a pool..."></textarea>
        </div>

        <button onclick="getRecommendations()">Find My Home</button>
        <div id="loading">🔍 Analyzing market data... Generating plots... Consulting Agent...</div>
        
        <div id="results" class="grid-results" style="display:none;">
            <div class="result-col" id="text-results">
                <!-- Text Cards go here -->
            </div>
            <div class="result-col" id="visual-results">
                <!-- Charts go here -->
            </div>
        </div>
    </div>

    <script>
        async function getRecommendations() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            const payload = {
                budget: document.getElementById('budget').value,
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseInt(document.getElementById('bathrooms').value),
                description: document.getElementById('description').value
            };

            try {
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function displayResults(data) {
            const textContainer = document.getElementById('text-results');
            const visualContainer = document.getElementById('visual-results');
            
            textContainer.innerHTML = '<h2>📝 AI Recommendations</h2>';
            visualContainer.innerHTML = '<h2>📊 Market Analysis</h2>';
            
            // 1. Process Text Data
            let recs = data.llm_data;
            if (typeof recs === 'string') {
                try { recs = JSON.parse(recs); } catch(e) { recs = null; }
            }
            
            if (recs && recs.recommendations) {
                recs.recommendations.forEach(rec => {
                    textContainer.innerHTML += `
                        <div class="card">
                            <h3>Property ID: ${rec['Property ID']}</h3>
                            <p>Match Score: <span class="score">${rec['Final Score']} / 100</span></p>
                            <p>${rec['Justification']}</p>
                        </div>
                    `;
                });
            } else {
                textContainer.innerHTML += `<div class="card">Raw Output: ${data.llm_data}</div>`;
            }

            // 2. Process Charts
            if (data.charts) {
                visualContainer.innerHTML += `
                    <div class="chart-container">
                        <h4>Match Score Comparison</h4>
                        <img src="data:image/png;base64,${data.charts.bar_chart}" />
                    </div>
                    <div class="chart-container">
                        <h4>Logic Breakdown (Heatmap)</h4>
                        <img src="data:image/png;base64,${data.charts.heatmap}" />
                    </div>
                `;
            }
            
            document.getElementById('results').style.display = 'grid';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_content

@app.post("/api/recommend")
async def get_recommendations(query: UserQuery):
    llm_raw, charts = agent.recommend_new_user(query)
    
    # Return a combined structure
    return {
        "llm_data": llm_raw,
        "charts": charts
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)