from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

app.secret_key = "house-price-prediction-secret"

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
features = saved_data["features"]
r2 = saved_data["r2_score"]


@app.route("/")
def home():
    history = session.get("history", [])
    return render_template("index.html", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    

    # Get values from website
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    mainroad = request.form["mainroad"]
    guestroom = request.form["guestroom"]
    basement = request.form["basement"]
    hotwaterheating = request.form["hotwaterheating"]
    airconditioning = request.form["airconditioning"]
    prefarea = request.form["prefarea"]
    furnishingstatus = request.form["furnishingstatus"]

    # Create input data
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad_yes": mainroad == "yes",
        "guestroom_yes": guestroom == "yes",
        "basement_yes": basement == "yes",
        "hotwaterheating_yes": hotwaterheating == "yes",
        "airconditioning_yes": airconditioning == "yes",
        "prefarea_yes": prefarea == "yes",
        "furnishingstatus_semi-furnished": furnishingstatus == "semi-furnished",
        "furnishingstatus_unfurnished": furnishingstatus == "unfurnished"
    }])

    # Match the exact feature order used during training
    input_data = input_data.reindex(
        columns=features,
        fill_value=False
    )

    # Predict price
    prediction = model.predict(input_data)[0]

    # Create dynamic prediction chart
    original_df = pd.read_csv("Housing.csv")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        original_df["area"],
        original_df["price"],
        alpha=0.5,
        label="Existing Houses"
    )

    plt.scatter(
        area,
        prediction,
        marker="*",
        s=250,
        label="Your Prediction"
    )

    plt.xlabel("Area (sq ft)")
    plt.ylabel("House Price")
    plt.title("Area vs House Price - Your Prediction")
    plt.legend()

    plt.savefig("static/prediction_chart.png")
    plt.close()

    # Format price
    prediction = f"{prediction:,.0f}"

    history = session.get("history", [])

    new_entry = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "prediction": prediction
    }

    if not history or history[-1] != new_entry:
        history.append(new_entry)

    session["history"] = history

    session["last_prediction"] = {
    "prediction": prediction,
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "r2_score": round(r2 * 100, 2)
}

    return render_template(
        "predict.html",
        prediction=prediction,
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        r2_score=round(r2 * 100, 2),
        history=history
    )

@app.route("/predict")
def predict_page():
    history = session.get("history", [])
    last_prediction = session.get("last_prediction")

    if last_prediction:
        return render_template(
            "predict.html",
            prediction=last_prediction["prediction"],
            area=last_prediction["area"],
            bedrooms=last_prediction["bedrooms"],
            bathrooms=last_prediction["bathrooms"],
            r2_score=last_prediction["r2_score"],
            history=history
        )

    return render_template(
        "predict.html",
        prediction=None,
        r2_score=round(r2 * 100, 2),
        history=history
    )


@app.route("/analysis")
def analysis_page():
    return render_template("analysis.html")


@app.route("/model")
def model_page():
    return render_template("model.html", r2_score=round(r2 * 100, 2))


@app.route("/history")
def history_page():
    history = session.get("history", [])
    return render_template("history.html", history=history)


@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/reset")
def reset():
    session.pop("last_prediction", None)
    return redirect(url_for("predict_page"))


if __name__ == "__main__":
    app.run(debug=True)