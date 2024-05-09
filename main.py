

"""This file using libraries"""
from flask import Flask, request, render_template
from src.model_test import get_prediction
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/visu/Artificial_Intelligence/IdentifyingHandwrittenMathematicalSymbols/testfile'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("ev.html")


@app.route('/read-form', methods=['POST']) 
def read_form(): 
  
    # Get the form data as Python ImmutableDict datatype  
    data = request.files['file']
    fil_path = secure_filename(data.filename)
    ## Return the extracted information 
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], fil_path)
    data.save(save_path)
    prediction_value = get_prediction(save_path)
    print(prediction_value)
    return f"<h4 style='text-align:center;color:#0000ff'>Class: {prediction_value}</h4>"

if __name__ == "__main__":
    app.run(debug=True)