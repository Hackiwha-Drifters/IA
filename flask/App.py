from flask import Flask, render_template, request, redirect , jsonify
from gevent.pywsgi import WSGIServer

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.layers import Dropout
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
import sklearn
from flask_cors import CORS

model = joblib.load('models/Bayes_model.pkl')

class_labels = ['Class_0', 'Class_1']

def predict_image_classification(img_path):
    img_array = preprocess_image_classification(img_path)
    predictions = classification_model.predict(img_array)
    predicted_probability = predictions[0][0]  # Assuming binary classification
    print(f'Predicted Probability: {predicted_probability}')
    
    # Using the threshold of 0.5 for binary classification
    if predicted_probability > 0.5:
        predicted_class = 1
        class_label = 'Yes Tumor'
    else:
        predicted_class = 0
        class_label = 'No Tumor'
        
    print(f'Predicted Class: {predicted_class} ({class_label})')
    
    return class_label, predicted_probability
# Function to preprocess the input image
def preprocess_image_classification(img_path):
    # Load the image using PIL
    img = Image.open(img_path)
    
    # Resize the image to the target size (64, 64)
    img = img.resize((64, 64))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array

# Function to preprocess the input image for segmentation
def preprocess_image_for_segmentation(img_path):
    # Load the image using PIL
    img = Image.open(img_path)
    
    # Resize the image to the target size (e.g., 256x256)
    img = img.resize((256, 256))
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Expand the dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    
    return img_array
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Define the dice coefficient metric function
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
    classification_model = load_model('models/classification_model.h5')
    segmentation_model = load_model('models/segmentation_model.h5')
def Segmentation_image (img_path):
    # Load and preprocess the test image
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256))  # Resize to match model input shape
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = segmentation_model.predict(img_array)

    # Post-process the predicted mask
    threshold = 0.5
    binary_mask = (predictions > threshold).astype(np.uint8)

    # Save the segmentation mask
    output_path = r'static/segmented_images/visualization.png'   
    plt.imsave(output_path, binary_mask[0, :, :, 0], cmap='gray')

    return output_path


app = Flask(__name__)
CORS(app)   

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/another_page')
def prediction():
    return render_template('prediction.html')

@app.route('/test',  methods=['GET'])
def test():
    return jsonify(test='bhb')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        img_path = os.path.join('static/images', file.filename)
        file.save(img_path)
         


        # Perform image segmentation
        predicted_class, confidence = predict_image_classification(img_path)
        print(predicted_class)

        
        segmented_image_path = Segmentation_image (img_path)
        # save_segmented_image(segmented_image_path, img_path)

        return render_template('index.html', predicted_class=predicted_class,image_path=img_path ,confidence=confidence ,segmented_image_path=segmented_image_path)

#heart desease prediction based on multiple factors 
@app.route('/predict', methods=['POST'])
def predict():
    print("je suis la ")
    data = request.json
    # Get the input data from the form
    feature1 = float(data['feature1'])
    feature2 = float(data['feature2'])
    feature3 = float(data['feature3'])
    feature4 = float(data['feature4'])
    feature5 = float(data['feature5'])
    feature6 = float(data['feature6'])
    feature7 = float(data['feature7'])
    feature8 = float(data['feature8'])
    feature9 = float(data['feature9'])
    feature10 = float(data['feature10'])
    feature11 = float(data['feature11'])
    feature12 = float(data['feature12'])
    feature13 = float(data['feature13'])
    
    print('je suis la ')
    # Use the loaded model to make predictions
    print(feature1)

    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13]])
    print("Prediction:", prediction[0])
    print('samiraTV')
    #return render_template('prediction.html', prediction=prediction[0])
    return jsonify(prediction=str(prediction[0]))

if __name__ == '__main__':
    # Debug/Development
    # app.run(debug=True, host="0.0.0.0", port="5000")
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    # app.run(host='0.0.0.0', port=5000, debug=True)
