# AutoXRD-Automated-XRD-Phase-Identification
AI-powered End-to-End Phase &amp; Composition Identification from XRD


Project Overview

AutoXRD is an end-to-end AI application that identifies crystalline phases and estimates approximate composition directly from raw 1D X-ray diffraction (XRD) data. It eliminates the need for manual baseline correction, peak picking, and phase matching, producing ready-to-view reports with annotated plots, phase abundances, and confidence scores.

Target Audience

Materials scientists and crystallographers (academic labs, industrial R&D)

High-throughput materials screening teams (battery, catalyst, metallurgy)

Service labs and quality control technicians

Need addressed: Fast, reproducible phase identification with uncertainty estimates and minimal manual processing.

Key Features

Single-click phase identification from raw 1D XRD data (supports multiple file formats).

Automated report generation (PDF/HTML) with annotated diffractograms and phase lists.

Mixture handling & uncertainty reporting (confidence and abundance estimates).

Batch processing and API access for integration with existing pipelines.

Technical Approach

Data: Labeled experimental datasets and simulated XRD patterns from crystallographic databases.

Modeling: 1D CNNs and Transformer encoders for multi-label phase identification and abundance regression.

Embedded Pre/Post-processing: Model robustness to baseline shifts, noise, and overlapping peaks; learnable denoising.

Auxiliary Modules: Composition estimator, database refinement, explainability via saliency maps.

Stack: PyTorch, pymatgen, ASE, FastAPI backend, React/Streamlit frontend, Docker deployment.

Evaluation: Precision/recall, F1-score, top-N accuracy, abundance mean absolute error (MAE).

Installation

Clone the repository:

git clone https://github.com/<username>/AutoXRD.git
cd AutoXRD


Install dependencies:

pip install -r requirements.txt


Start the application:

streamlit run app.py
# or for API:
uvicorn main:app --reload

Usage

Upload raw 1D XRD data (CSV, TXT, or other supported formats).

Click Analyze to perform automated phase identification.

View annotated diffractogram with predicted phases and confidence scores.

Export results as PDF or HTML report.

Example Reports

Single-phase identification

Two-phase mixture analysis

Noisy XRD pattern handling

(PDF examples available in /reports folder)

Evaluation Metrics

Phase Identification: Precision, Recall, F1-score, Top-N accuracy

Abundance Estimation: Mean Absolute Error (MAE)

Robustness: Performance across noisy, shifted, or instrument-diverse datasets

Challenges & Mitigations

Limited labeled mixtures: Simulations, data augmentation, active learning

Peak overlap & polymorph discrimination: High-capacity models, physics-informed constraints

Instrument variability: Domain adaptation, diverse training sets

Trust & interpretability: Confidence scores, attribution plots, raw matching scores

Future Plans

Integration with LIMS

Retraining with user feedback

Lattice refinement & Rietveld suggestions

SaaS or on-prem Docker deployment

Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or dataset additions.

License

MIT License
