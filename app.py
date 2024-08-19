from flask import Flask, render_template, request, jsonify,redirect
import redis
import json
from main_srk import SRK
from main_osrk import OSRK
from binit import bindf
from preprocess import pre
import pickle
import pandas as pd
import pickle
import io
import time

app = Flask(__name__)
file_df = False
file_csv = False
model_file = False
res_dict = False
last_column = False

r = redis.Redis(
host='redis-13060.c61.us-east-1-3.ec2.redns.redis-cloud.com',
port=13060,
password='LRbEOTaAJ79okUuwFP7l0qIlycxfQuz0')

models = False

pred_data_df = False

def predict_df(dataset_key,model_key,file = False):

    if model_key == "null":
        model = pickle.loads(file)
        print(type(model))

    else:
        model = models[dataset_key][model_key]
        model = pickle.loads(model)

    data_df = json.loads(r.hget(dataset_key,"test"))
    train_df = pd.DataFrame.from_dict(data_df)

    data_df = train_df.reset_index(drop=True)
    X_train = data_df.iloc[:, :-1]
    y_pred = model.predict(X_train)

    # Convert NumPy array to Pandas Series
    y_pred_series = pd.Series(y_pred, name='pred_target')
    new_data_df = pd.concat([X_train, y_pred_series], axis=1)

    return new_data_df

# Function to fetch all records from Redis and parse them
def fetch_all_records():
    global models
    # Fetch all keys
    all_keys = r.keys('*')

    datasets = list()

    models = {}
    model_keys = {}

    # Iterate over keys and fetch their values
    for key in all_keys:
        key_name = key.decode('utf-8')  # Convert bytes to string
        
        hash_values = r.hgetall(key_name)

        model_keys[key_name] = []
        models[key_name] = {}

        for item in hash_values.keys():
            hash = item.decode('utf-8')
            if "Model" in hash:
                model_keys[key_name].append(hash)
                models[key_name][hash] = r.hget(key_name,hash)

        datasets.append({'name': key_name, 'num_m': len(models[key_name])})
    
    return datasets,model_keys



# Dummy model function
def run_model(method, data_df,epsilon,num_samples,res_dict = False):

    # Retrieve serialized model from Redis
    ##serialized_model = r.hget(datakey, "model") ########### FOR SRK

    # Deserialize the model
    ##model = pickle.loads(serialized_model)   ############## FOR SRK

    # Replace this with your actual model logic
    if method == 'SRK':
        results = SRK(res_dict,data_df,epsilon,num_samples)

    elif method == 'OSRK':
        results = OSRK(data_df,epsilon,num_samples)

    #elif method == 'DYN':

    return results

@app.route('/')
def index():
    global file_df,file_csv,new_data_df,model_file,res_dict, last_column
    file_df,file_csv,new_data_df,model_file,res_dict,last_column = False,False, False, False, False, False
    datasets,model_keys = fetch_all_records()
    print(model_keys)
    return render_template('index.html', datasets=datasets, models = model_keys, dataset = None, model = None, pred_df = None, pred = None)


@app.route('/modelFile',methods = ['POST'])
def modelFile():
    global model_file

    model_file = request.files['file'].read()

    print(type(model_file))

    print("OK")

    return "OK"

@app.route('/predict')
def predict():
    global pred_data_df, model_file, res_dict, last_column

    print(type(model_file))
    

    file = model_file

    dataset_key = request.args.get('dataset')
    model_key = request.args.get('model')

    # Run your model with the provided parameters
    pred = predict_df(dataset_key,model_key,file)

    pred_data_df = pred

    n = len(pred)

    csv_data = r.hget(dataset_key,"csv")

    res_dict = pickle.loads(r.hget(dataset_key,"res_dict"))

    csv_content = csv_data.decode('utf-8')

    data_df = pd.read_csv(io.StringIO(csv_content))

    last_column = data_df.columns.values.tolist()[-1]

    print(last_column)

    data_df = data_df.reset_index(drop=True)
    
    X = data_df.iloc[:n, :-1]
    y = pred.iloc[:, -1:]

    new_data_df = pd.concat([X, y], axis=1)

    df_html = new_data_df.to_html(classes='dataframe', index=False)
    
    return render_template('index.html', datasets=None, models = None, pred = df_html, dataset = dataset_key, model = model_key)


@app.route('/results')
def results():
    method = request.args.get('method')
    epsilon = request.args.get('epsilon', type=float)
    num_samples = request.args.get('num_samples', type=int)

    # Run your model with the provided parameters
    results = run_model(method, pred_data_df,epsilon,num_samples, res_dict)

    # Convert DataFrame to HTML
    df_html = results["res_df"].to_html(classes='table table-striped', index=False)
    
    return render_template('results.html', results=results, table=df_html)


@app.route('/data')
def data():
    global file_df,file_csv

    file_df = False
    file_csv = False

    datasets,model_keys = fetch_all_records()
    return render_template('data.html', datasets=datasets, models = model_keys)

@app.route('/process', methods=['POST'])
def process_file():

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        global file_df,file_csv
        df = pd.read_csv(file, index_col=0)

        file_csv = df
        
        # First processing function
        df = bindf(df)

        file_df = df
        
        # Convert DataFrame to HTML table and render template
        return render_template('data.html', results=df.to_html(classes='table table-bordered', index=False))
    
    return redirect(request.url)


@app.route('/upload', methods=['POST'])
def upload():
    
    global file_df

    num_models = int(request.form.get('num_models', 5))
    file = file_df

    multiclass= request.form.get('mutli', False)
    name = request.form.get('name')

    csv_buffer = io.StringIO()
    file_csv.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    r.hset(name, "csv", csv_buffer.getvalue())


    if multiclass == "Yes":
        multiclass = True

    else:
        multiclass = False
   
    best_model, models, train, test,res_dict = pre(file,multiclass,num_models)

    if not (best_model and name):
        # Convert DataFrame to HTML table and render template
        return render_template('data.html', results=file.to_html(classes='table table-bordered', index=False))

    else:
        print("SUCCESS") #REDIS DATA UPLOAD
        r.hset(name, "test", test.to_json())
        r.hset(name, "train", train.to_json())
        r.hset(name,"res_dict",pickle.dumps(res_dict))
        r.hset(name,"Best Model",best_model)

        for i in range(0,len(models)):
            r.hset(name,f"Model {i+1}",models[i])
            
        datasets,model_keys = fetch_all_records()
    
        return render_template('data.html', success = True , datasets=datasets, models = model_keys)

    
if __name__ == '__main__':

    app.run(debug=True)
