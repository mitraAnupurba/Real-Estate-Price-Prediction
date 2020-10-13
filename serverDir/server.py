from flask import Flask,request,jsonify
import serverDir.util as util
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hi"

@app.route('/get_all_locations')
def get_all_locations():
    response = jsonify({
        'locations' : util.get_all_locations()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response
    return 'Hi'


@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = float(request.form['location'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price' : util.get_estimated_price(location,total_sqft,bath,bhk)
    })

    return response

if __name__ == "__main__":
    print("starting the flask serverDir for home price prediction")
    app.run()