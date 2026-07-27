from flask import Flask, render_template, request, send_from_directory, session, send_file
import joblib
import pandas as pd
import os
import shap
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-key-change-later'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'xgboost.pkl')
    model = joblib.load(MODEL_PATH)

    LASSO_PATH = os.path.join(BASE_DIR, '..', 'data', 'X_lasso.csv')
    expected_probes = pd.read_csv(LASSO_PATH, index_col=0).drop(columns=['label'], errors='ignore').columns.tolist()

    lasso_df = pd.read_csv(LASSO_PATH, index_col=0)
    if 'label' in lasso_df.columns:
        lasso_df = lasso_df.drop(columns=['label'])
    probe_means = lasso_df[expected_probes].mean()

    # Create SHAP explainer once at startup — TreeExplainer is fast per-sample
    explainer = shap.TreeExplainer(model)

    @app.route('/')
    def landing():
        return render_template('landing.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if request.method == 'POST':
            file = request.files.get('csv_file')
            if not file or file.filename == '':
                return "No file uploaded — please choose a CSV file."

            df = pd.read_csv(file)

            missing = set(expected_probes) - set(df.columns)
            if missing:
                return f"CSV is missing {len(missing)} required probe columns. First few missing: {list(missing)[:5]}"

            X = df[expected_probes]

            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities)
            label = "Tumor" if prediction == 1 else "Non-Tumor"

            sample_values = X.iloc[0]
            probe_strip = [
                "high" if sample_values[probe] >= probe_means[probe] else "low"
                for probe in expected_probes
            ]

            # SHAP values for this single sample
            shap_values = explainer.shap_values(X)
            sample_shap = shap_values[0]  # shape: (71,)

            # Pair each probe with its SHAP value, sort by absolute impact
            gene_impacts = list(zip(expected_probes, sample_shap))
            gene_impacts.sort(key=lambda pair: abs(pair[1]), reverse=True)
            top_genes = gene_impacts[:5]

            top_genes_display = [
                {
                    "probe": probe,
                    "direction": "toward Tumor" if val > 0 else "toward Non-Tumor",
                    "magnitude": float(abs(val))
                }
                for probe, val in top_genes
            ]

            # Stash results in session for PDF export
            session['last_result'] = {
                'label': label,
                'confidence': f"{confidence:.1%}",
                'top_genes': top_genes_display
            }

            return render_template(
                'results.html',
                label=label,
                confidence=f"{confidence:.1%}",
                probe_strip=probe_strip,
                top_genes=top_genes_display
            )

        return render_template('predict.html')
    @app.route('/static-sample')
    def static_sample():
        sample_dir = os.path.join(BASE_DIR, 'sample_data')
        return send_from_directory(sample_dir, 'sample_input.csv')
    @app.route('/download-pdf')
    def download_pdf():
        result = session.get('last_result')
        if not result:
            return "No recent prediction found. Please run a prediction first."

        from reportlab.lib.colors import HexColor

        # Design tokens, matching the web app
        INK = HexColor('#0F2733')
        INK_SOFT = HexColor('#3E5C66')
        TEAL = HexColor('#0F766E')
        CORAL = HexColor('#E4572E')
        BORDER = HexColor('#D8E0E2')
        PAPER_WHITE = HexColor('#FFFFFF')

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        prediction_color = CORAL if result['label'] == 'Tumor' else TEAL

        # --- Header bar ---
        c.setFillColor(TEAL)
        c.rect(0, height - 1.3 * inch, width, 1.3 * inch, fill=1, stroke=0)

        c.setFillColor(PAPER_WHITE)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(1 * inch, height - 0.7 * inch, "HBV-HCC Prediction Report")

        c.setFont("Helvetica", 9)
        c.drawString(1 * inch, height - 1.0 * inch,
                     f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}")

        y = height - 1.8 * inch

        # --- Prediction section ---
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica", 9)
        c.drawString(1 * inch, y, "PREDICTION")

        c.setFillColor(prediction_color)
        c.setFont("Helvetica-Bold", 28)
        c.drawString(1 * inch, y - 0.45 * inch, result['label'])

        # Confidence, right-aligned in the same row
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica", 9)
        c.drawString(4.5 * inch, y, "CONFIDENCE")

        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 28)
        c.drawString(4.5 * inch, y - 0.45 * inch, result['confidence'])

        y -= 0.9 * inch

        # Divider line
        c.setStrokeColor(BORDER)
        c.setLineWidth(1)
        c.line(1 * inch, y, width - 1 * inch, y)

        y -= 0.45 * inch

        # --- Top contributing genes section ---
        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica", 9)
        c.drawString(1 * inch, y, "TOP CONTRIBUTING GENES")

        y -= 0.35 * inch

        for i, gene in enumerate(result['top_genes']):
            row_color = TEAL if 'Non-Tumor' in gene['direction'] else CORAL

            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(1.1 * inch, y, gene['probe'])

            c.setFillColor(row_color)
            c.setFont("Helvetica", 11)
            c.drawRightString(width - 1.1 * inch, y, gene['direction'])

            y -= 0.15 * inch
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.5)
            c.line(1 * inch, y, width - 1 * inch, y)

            y -= 0.3 * inch

        # --- Footer ---
        c.setStrokeColor(BORDER)
        c.line(1 * inch, 0.9 * inch, width - 1 * inch, 0.9 * inch)

        c.setFillColor(INK_SOFT)
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(1 * inch, 0.7 * inch,
                     "Demo only. Not for clinical use.")

        c.save()
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='hbv_hcc_prediction_report.pdf',
            mimetype='application/pdf'
        )
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)