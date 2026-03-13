/* website/demo.js — QuantumPINNs interactive demo logic */

const API_BASE = "http://localhost:5000";

async function submitPrediction() {
  const xInput = document.getElementById("x-input").value.trim();
  const tInput = document.getElementById("t-input").value.trim();
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");

  if (!xInput) {
    statusEl.textContent = "Please enter x values.";
    return;
  }

  const xArr = xInput.split(",").map(Number).filter(v => !isNaN(v));
  const tArr = tInput
    ? tInput.split(",").map(Number).filter(v => !isNaN(v))
    : new Array(xArr.length).fill(0);

  if (xArr.length !== tArr.length) {
    statusEl.textContent = "x and t arrays must have the same length.";
    return;
  }

  statusEl.textContent = "Running inference...";
  resultEl.textContent = "";

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: xArr, t: tArr }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      statusEl.textContent = `Error: ${err.error || resp.statusText}`;
      return;
    }

    const data = await resp.json();
    statusEl.textContent = "Done.";

    const lines = xArr.map((x, i) => {
      const psiR = data.psi_real[i].toFixed(6);
      const psiI = data.psi_imag ? data.psi_imag[i].toFixed(6) : "N/A";
      const prob = data.prob_density[i].toFixed(6);
      return `x=${x.toFixed(3)}  t=${tArr[i].toFixed(3)}  ψ_r=${psiR}  ψ_i=${psiI}  |ψ|²=${prob}`;
    });
    resultEl.textContent = lines.join("\n");
  } catch (e) {
    statusEl.textContent = `Network error: ${e.message}. Is the API server running?`;
  }
}

async function checkHealth() {
  const el = document.getElementById("health-status");
  try {
    const resp = await fetch(`${API_BASE}/health`);
    const data = await resp.json();
    el.textContent = data.model_loaded
      ? "API online — model loaded"
      : "API online — no model loaded";
    el.style.color = data.model_loaded ? "#3fb950" : "#ffa657";
  } catch {
    el.textContent = "API offline";
    el.style.color = "#ff7b72";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setInterval(checkHealth, 10000);
});
