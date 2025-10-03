AutoXRD

AI-powered End-to-End Phase & Composition Identification from XRD

From raw XRD patterns to actionable phase and composition insights — fully automated.

📌 Project Overview

AutoXRD is an AI application that identifies crystalline phases and estimates approximate compositions directly from raw 1D X-ray diffraction (XRD) data.

It eliminates tedious manual pre/post-processing (baseline correction, smoothing, peak picking, phase matching), and generates ready-to-use reports with:

Annotated diffractograms

Phase list with probabilities

Estimated abundances

Confidence scores

🎯 Target Audience

Materials scientists & crystallographers (academic labs, industrial R&D)

High-throughput screening teams (batteries, catalysts, metallurgy)

QC/Service labs (fast phase ID, reproducible reports)

🚀 Key Features

✅ Single-click phase identification from raw 1D XRD data

✅ Automated report generation (PDF/HTML)

✅ Mixture handling & uncertainty estimates

✅ Batch processing & API access

✅ Integration with existing pipelines

🛠️ Tech Stack

ML/DL: PyTorch (1D CNN / Transformer encoders)

Materials Science: pymatgen, ASE

Backend/API: FastAPI

Frontend: Streamlit / React

Deployment: Docker

📂 Repository Structure
AutoXRD/
│
├── data/               # Example dataset (sample_xrd.csv)
├── models/             # Saved model weights
├── reports/            # Example generated reports
├── xrd_model.py        # CNN model for phase & abundance prediction
├── train.py            # Training loop (dummy dataset included)
├── app.py              # Streamlit demo for inference
├── utils.py            # Helper functions (plotting, etc.)
├── requirements.txt    # Dependencies
├── README.md           # Project description
├── .gitignore
└── LICENSE

⚡ Installation

Clone the repo:

git clone https://github.com/<username>/AutoXRD.git
cd AutoXRD


Install dependencies:

pip install -r requirements.txt

▶️ Usage
Train the model
python train.py

Run the demo app
streamlit run app.py


Upload a CSV file with columns:

theta,intensity


You’ll see:

XRD diffractogram (plotted)

Predicted phases (probabilities)

Estimated abundances

📊 Example Output

Predicted Phases (probabilities):

[0.82, 0.12, 0.05, 0.01, 0.03]


Predicted Abundances:

[0.60, 0.20, 0.10, 0.05, 0.05]

📈 Evaluation Metrics

Phase ID: Precision, Recall, F1-score, Top-N accuracy

Abundance estimation: Mean Absolute Error (MAE)

Robustness: Performance across noisy & shifted data

🔮 Future Roadmap

Integration with LIMS

User feedback retraining loop

Rietveld refinement suggestions

SaaS + Dockerized deployment

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

📜 License

This project is licensed under the MIT License.
