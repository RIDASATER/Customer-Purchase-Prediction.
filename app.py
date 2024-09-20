import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the trained DecisionTreeClassifier model
model = pickle.load(open('DataSet_predection.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if request.json and 'data' in request.json:
        data = request.json['data']
        print(data)
        
        # Convert input data to numpy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        
        # Predict using the model
        output = model.predict(input_data)
        
        return jsonify(output[0])  # Return the predicted value
    else:
        return jsonify({"error": "Invalid input, 'data' key is missing"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    
    # Convert input data to numpy array
    final_input = np.array(data).reshape(1, -1)
    
    print(final_input)
    
    # Predict using the model
    output = model.predict(final_input)[0]
    
    return render_template('home.html', prediction_text="The predicted value is : {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
