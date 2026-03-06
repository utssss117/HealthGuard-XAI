/* ──────────────────────────────────────────────────────────────────────────
   app.js  —  HealthGuard XAI Dashboard
   Connects to FastAPI backend at http://localhost:8000
────────────────────────────────────────────────────────────────────────── */

const API = "http://localhost:8000";

// ── State ──────────────────────────────────────────────────────────────────
let lastPrediction  = null;   // { risk_probability, risk_level, top_features }
let lastBiomarkers  = null;   // raw form object
let chatHistory     = [];

// ── Tab Switching ──────────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active") || t.classList.add("hidden"));
    btn.classList.add("active");
    const tab = document.getElementById(`tab-${btn.dataset.tab}`);
    tab.classList.remove("hidden");
    tab.classList.add("active");
  });
});

// ── Form Submit → Predict ──────────────────────────────────────────────────
document.getElementById("predict-form").addEventListener("submit", async e => {
  e.preventDefault();
  const fields = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"];
  const body   = Object.fromEntries(fields.map(f => [f, parseFloat(document.getElementById(f).value)]));

  showLoader("Analyzing patient data…");
  try {
    const r   = await fetch(`${API}/predict`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Prediction failed");

    lastPrediction = data;
    lastBiomarkers = body;

    renderResults(data);
    document.getElementById("results-section").classList.remove("hidden");

    // Scroll to results
    setTimeout(() => document.getElementById("results-section").scrollIntoView({behavior:"smooth"}), 100);

    // Fire recommendations in background
    loadRecommendations(body, data.risk_probability);

  } catch(err) { alert("Error: " + err.message); }
  finally { hideLoader(); }
});

// ── Render Risk Gauge ──────────────────────────────────────────────────────
function renderResults(data) {
  const pct = data.risk_probability;
  setGauge(pct);
  renderBadge(data.risk_level);
  renderFeatureChart(data.top_features);
}

function setGauge(prob) {
  const arc     = document.getElementById("gauge-arc");
  const needle  = document.getElementById("gauge-needle");
  const pctEl   = document.getElementById("gauge-pct");

  const total   = 251.2;                         // half-circle circumference
  const offset  = total - total * prob;
  const angle   = -90 + prob * 180;             // -90° (left) to +90° (right)

  arc.style.strokeDashoffset    = offset;
  needle.setAttribute("transform", `rotate(${angle}, 100, 100)`);
  pctEl.textContent             = Math.round(prob * 100) + "%";

  // Color the needle based on risk
  const color = prob <= 0.33 ? "#22c55e" : prob <= 0.66 ? "#f59e0b" : "#ef4444";
  needle.setAttribute("stroke", color);
}

function renderBadge(level) {
  const b = document.getElementById("risk-badge");
  b.textContent = level + " Risk";
  b.className   = "risk-badge " + level;
}

// ── Feature Importance Chart ───────────────────────────────────────────────
function renderFeatureChart(features) {
  const container = document.getElementById("feature-chart");
  container.innerHTML = "";
  const entries = Object.entries(features).slice(0, 8);
  const max     = Math.max(...entries.map(([,v]) => Math.abs(v)), 0.001);

  entries.forEach(([name, val]) => {
    const pct  = Math.abs(val) / max * 100;
    const row  = document.createElement("div");
    row.className = "feat-row";
    row.innerHTML = `
      <span class="feat-name">${name}</span>
      <div class="feat-bar-bg">
        <div class="feat-bar" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="feat-val">${val.toFixed(3)}</span>
    `;
    container.appendChild(row);
  });

  // Animate bars after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".feat-bar").forEach(bar => {
      bar.style.width = bar.dataset.target + "%";
    });
  });
}

// ── Recommendations ────────────────────────────────────────────────────────
async function loadRecommendations(biomarkers, riskProb) {
  const list = document.getElementById("recs-list");
  list.innerHTML = `<div class="loader">Generating recommendations…</div>`;

  const predicted_risks = { diabetes: riskProb, heart_disease: riskProb * 0.8 };
  const payload = {
    biomarkers,
    predicted_risks,
    top_positive_risk_factors: [],
    protective_factors: [],
    use_llm: false,
  };

  try {
    const r    = await fetch(`${API}/recommend`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail);

    list.innerHTML = "";
    const recs = data.prioritized_recommendations || [];
    recs.slice(0, 5).forEach((rec, i) => {
      const div = document.createElement("div");
      div.className = "rec-item";
      div.style.animationDelay = `${i * 80}ms`;
      div.innerHTML = `
        <span class="rec-priority">#${i+1}</span>
        <span class="rec-text">${rec.recommendation}
          <span class="rec-tag">${rec.related_risk || "general"}</span>
        </span>
      `;
      list.appendChild(div);
    });

    if (!recs.length) list.innerHTML = `<div class="loader">No specific recommendations generated.</div>`;

  } catch(err) {
    list.innerHTML = `<div class="loader">Could not load recommendations: ${err.message}</div>`;
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────
document.getElementById("chat-send").addEventListener("click", sendChat);
document.getElementById("chat-input").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

async function sendChat() {
  const input  = document.getElementById("chat-input");
  const text   = input.value.trim();
  if (!text) return;
  input.value  = "";

  appendMsg("user", text);
  chatHistory.push({ role:"user", content:text });

  // Build patient_data from last prediction (if available)
  const patient_data = lastPrediction ? {
    predicted_risks:  { diabetes: lastPrediction.risk_probability, heart_disease: lastPrediction.risk_probability * 0.8 },
    risk_level:       lastPrediction.risk_level,
    top_risk_factors: Object.keys(lastPrediction.top_features).slice(0, 3),
    protective_factors: [],
    patient_profile:  lastBiomarkers || {},
  } : { predicted_risks:{}, risk_level:"Unknown", top_risk_factors:[], protective_factors:[], patient_profile:{} };

  const typingEl = appendMsg("bot", "…", true);

  try {
    const r    = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ message:text, patient_data, history: chatHistory.slice(-6) }),
    });
    const data = await r.json();
    typingEl.remove();

    const reply = data.assistant_response || "Sorry, I couldn't process that.";
    appendMsg("bot", reply);
    chatHistory.push({ role:"assistant", content:reply });

    if (data.safety_flag) {
      appendMsg("bot", "⚠️ Safety notice: Please consult a healthcare professional for clinical decisions.");
    }
  } catch(err) {
    typingEl.remove();
    appendMsg("bot", `Connection error: ${err.message}. Make sure the API server is running.`);
  }
}

function appendMsg(role, text, isTyping=false) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="msg-bubble${isTyping ? " typing" : ""}">${escHtml(text)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function escHtml(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\n/g,"<br/>");
}

// ── Loader ─────────────────────────────────────────────────────────────────
function showLoader(msg="Loading…") {
  document.getElementById("loader-text").textContent = msg;
  document.getElementById("loader-overlay").classList.remove("hidden");
}
function hideLoader() {
  document.getElementById("loader-overlay").classList.add("hidden");
}
