"""
server.py — Flask inference API for QuantumPINNs.

Endpoints:
    GET  /health            Liveness check.
    POST /predict           Run trained PINN on supplied (x, t) inputs.
    GET  /problems          List available problem types.

Usage:
    python -m src.server --model-path model.pt --problem harmonic_oscillator
"""

import argparse
import json
import os

import numpy as np
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.pinn import PINN, ComplexPINN

app = Flask(__name__)
CORS(app)

# Global model state (loaded at startup)
_state: dict = {
    "model": None,
    "problem": None,
    "device": "cpu",
}


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _state["model"] is not None})


@app.get("/problems")
def problems():
    return jsonify({
        "problems": [
            "harmonic_oscillator",
            "schrodinger",
            "anharmonic",
            "hamiltonian",
        ]
    })


@app.post("/predict")
def predict():
    """Evaluate the loaded PINN on provided inputs.

    Request body (JSON):
        {
            "x": [float, ...],       // spatial coordinates
            "t": [float, ...],       // time coordinates (same length as x)
            "problem": "harmonic_oscillator"   // optional override
        }

    Response (JSON):
        {
            "psi_real":    [float, ...],
            "psi_imag":    [float, ...],   // null for real-valued problems
            "prob_density": [float, ...]
        }
    """
    if _state["model"] is None:
        return jsonify({"error": "No model loaded. Start the server with --model-path."}), 503

    body = request.get_json(force=True)
    if not body or "x" not in body:
        return jsonify({"error": "Request body must contain 'x' array."}), 400

    x_vals = body["x"]
    t_vals = body.get("t", [0.0] * len(x_vals))

    if len(x_vals) != len(t_vals):
        return jsonify({"error": "'x' and 't' must have the same length."}), 400
    if len(x_vals) > 10_000:
        return jsonify({"error": "Maximum 10,000 points per request."}), 400

    device = _state["device"]
    x = torch.FloatTensor(x_vals).unsqueeze(1).to(device)
    t = torch.FloatTensor(t_vals).unsqueeze(1).to(device)

    model = _state["model"]
    model.eval()

    with torch.no_grad():
        if isinstance(model, ComplexPINN):
            psi_r, psi_i = model(x, t)
            prob = (psi_r**2 + psi_i**2).cpu().numpy().flatten().tolist()
            result = {
                "psi_real":    psi_r.cpu().numpy().flatten().tolist(),
                "psi_imag":    psi_i.cpu().numpy().flatten().tolist(),
                "prob_density": prob,
            }
        else:
            psi = model(x, t)
            result = {
                "psi_real":    psi.cpu().numpy().flatten().tolist(),
                "psi_imag":    None,
                "prob_density": (psi**2).cpu().numpy().flatten().tolist(),
            }

    return jsonify(result)


# ── Model loader ───────────────────────────────────────────────────────────


def load_model(model_path: str, problem: str, device: str) -> None:
    checkpoint = torch.load(model_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    if problem == "schrodinger":
        model = ComplexPINN(
            in_dim=saved_args.get("in_dim", 2),
            hidden_dim=saved_args.get("hidden_dim", 80),
            n_layers=saved_args.get("n_layers", 5),
            activation=saved_args.get("activation", "tanh"),
        )
    else:
        model = PINN(
            in_dim=saved_args.get("in_dim", 2),
            out_dim=1,
            hidden_dim=saved_args.get("hidden_dim", 64),
            n_layers=saved_args.get("n_layers", 4),
            activation=saved_args.get("activation", "tanh"),
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _state["model"]   = model
    _state["problem"] = problem
    _state["device"]  = device
    print(f"[server] model loaded from {model_path}  problem={problem}  device={device}")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantumPINNs inference server.")
    p.add_argument("--model-path", default=None,
                   help="Path to a saved model checkpoint (.pt).")
    p.add_argument("--problem", default="harmonic_oscillator",
                   help="Problem type matching the trained model.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if args.model_path and os.path.isfile(args.model_path):
        load_model(args.model_path, args.problem, device)
    else:
        print("[server] No model path provided or file not found — running without a loaded model.")

    app.run(host=args.host, port=args.port, debug=False)
