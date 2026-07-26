from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import shap
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
                    "magnitude": abs(val)
                }
                for probe, val in top_genes
            ]

            return render_template(
                'results.html',
                label=label,
                confidence=f"{confidence:.1%}",
                probe_strip=probe_strip,
                top_genes=top_genes_display
            )

        return render_template('predict.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)