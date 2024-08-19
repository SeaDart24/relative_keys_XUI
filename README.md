# relative_keys_XUI

This is an extension to the work by Shuai An on the Relative keys algorithm. https://github.com/shuaianuoe/relative_keys/tree/main 

This work modifies and further adds to his work by making it suitable to an online and dynamic setting that is user interactable and simple. This README provides instructions for running the Flask app, uploading datasets, and querying explanations. The application allows you to interact with machine learning models, upload datasets, and retrieve explanations for predictions.


Firstly, the following packages are necessary:
```
numpy 1.20.3
pandas 2.0.1
scikit-learn 0.24.2
xgboost 1.7.1
redis 4.6.0
flask 2.8.1
```

### 1 To start the Flask application

1. Run the Flask app by executing the following command in your terminal:

```
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000` to view the application.

3. Follow the on-screen instructions to interact with the app.


### To upload datasets

1. Go to `http://localhost:5000/data/`

2. Follow the on-screen instructions to upload your datasets.


### Query and Get Explanations

1. `Navigate to http://localhost:5000/`

2. Select the uploaded dataset and the corresponding model from the available options.

3. If needed, upload a compatible model for the dataset in `.pkl` format.


### Sample Data and Models

Sample datasets used in the project are provided in the `/data` folder.

A sample model for the partial_loan dataset is available in the `/models` folder for testing purposes.

### Additional Information
Ensure that all necessary Python packages and dependencies are installed.

For further details and usage, refer to the in-app guidance and prompts.
