# Social Media Usage & Student Mental Health

Streamlit app for the Advanced Data Science FA2 case study. It estimates a student's mental health score from survey
inputs using trained regression models. Predictions show patterns in the survey data; they do not establish causes and
are not a medical assessment.

## Run locally

From the project root, install the dependencies and start the packaged app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_deploy/streamlit_deploy/app.py
```

The app and trained models are in `streamlit_deploy/streamlit_deploy/`.

## Deploy with Streamlit Community Cloud

Connect this GitHub repository and set **Main file path** to `app.py` in the repository root. The root app loads the
trained models from `streamlit_deploy/streamlit_deploy/models/`. Set the app's requirements file to `requirements.txt`
in the repository root.

## Project files

- `ADS_FA2_Case_Study.ipynb`: case study notebook.
- `Students_Social_Media_Addiction.csv`: survey dataset.
- `streamlit_deploy/streamlit_deploy/app.py`: interactive prediction app.
- `streamlit_deploy/streamlit_deploy/models/`: trained models and model metadata.
