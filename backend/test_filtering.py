import requests
import glob
import time

csvs = glob.glob('.storage/uploads/*tamilnadu*.csv')
if not csvs:
    csvs = glob.glob('.storage/uploads/*.csv')
    print('Tamil Nadu CSV not found, using fallback:', csvs[0])

if csvs:
    url = 'http://localhost:8000/pipeline/run'
    with open(csvs[0], 'rb') as f:
        files = {'file': ('election.csv', f, 'text/csv')}
        data = {
            'goal': 'Predict % Votes',
            'target_col': '% Votes',
            'selected_models': '["LogisticRegression", "RandomForestClassifier"]',
            'fast_mode': 'True',
            'run_explainability': 'True'
        }
        print("Sending intentionally wrong classification models for Regression task...")
        resp = requests.post(url, files=files, data=data)
        print('POST /pipeline/run Status Code:', resp.status_code)
        if resp.status_code == 200:
            res_data = resp.json()
            run_id = res_data['run_id']
            print('Run ID:', run_id)
            print('Detected Task Type:', res_data.get('task_type'))
            print('Filtered Models:', res_data.get('filtered_models'))
            
            for i in range(25):
                time.sleep(3)
                status_resp = requests.get('http://localhost:8000/pipeline/status/' + run_id)
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    print(f"Status: {status_data.get('status')} | Progress: {status_data.get('progress_pct')}%")
                    if status_data.get('status') in ['completed', 'failed', 'error']:
                        break
else:
    print('No CSV found.')
