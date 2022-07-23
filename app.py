import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def load_model():
    with open("model_saved.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()

model = data["model"]
leJobTitle = data["leJobTitle"]
leCompanyName = data["leCompanyName"]
leLocation = data["leLocation"]
leSize = data["leSize"]
leTypeeOfOwnership = data["leTypeeOfOwnership"]
leIndustry = data["leIndustry"]
leSector = data["leSector"]
leRevenue = data["leRevenue"]


@app.route("/")
def home():
    return render_template("index.html")


# This is from where our model predicts the output
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print(data)
    values = {}
    for k, v in data.items():
        values[str(k)] = str(v)
    temp = []
    temp.append(leJobTitle.transform([values["title"]]))
    temp.append(values["rating"])
    temp.append(leCompanyName.transform([values["company"]]))
    temp.append(leLocation.transform([values["location"]]))
    temp.append(leSize.transform([values["size"]]))
    temp.append(leTypeeOfOwnership.transform([values["ownerShip"]]))
    temp.append(leIndustry.transform([values["industry"]]))
    temp.append(leSector.transform([values["sector"]]))
    temp.append(leRevenue.transform([values["revenue"]]))
    temp.append(values["python"])
    temp.append(values["excel"])
    temp.append(values["hadoop"])
    temp.append(values["spark"])
    temp.append(values["aws"])
    temp.append(values["tableau"])
    temp.append(values["big_data"])
    prediction = model.predict([temp])
    return (
        jsonify(
            {"min": prediction[0][0], "max": prediction[0][1], "avg": prediction[0][2]}
        ),
        201,
    )


if __name__ == "__main__":
    app.run(debug=True)

#
