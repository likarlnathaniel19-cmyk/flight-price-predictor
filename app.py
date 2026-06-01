from flask import Flask, request, render_template
import pickle
import pandas as pd
from geopy.distance import geodesic

app = Flask(__name__)

ASIAN_GEO_DATABASE = {
    'Manila (MNL)': (14.5995, 120.9842),
    'Cebu (CEB)': (10.3157, 123.8854),
    'Singapore (SIN)': (1.3521, 103.8198),
    'Tokyo (HND/NRT)': (35.6764, 139.6500),
    'Seoul (ICN)': (37.5665, 126.9780),
    'Bangkok (BKK)': (13.7563, 100.5018),
    'Taipei (TPE)': (25.0330, 121.5654),
    'Hong Kong (HKG)': (22.3193, 114.1694),
    'Kuala Lumpur (KUL)': (3.1390, 101.6869),
    'Jakarta (CGK)': (-6.2088, 106.8456),
    'New Delhi (DEL)': (28.6139, 77.2090)
}

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("ASIAN REGIONAL ENGINE: Online & Deployable.")
except Exception as e:
    print(f"BOOT SYSTEM FAULT: Run train.py first to build model.pkl. Error: {e}")
    model = None

@app.route("/")
def home():
    return render_template(
        "index.html", 
        selected_airline="Cebu Pacific", 
        selected_source="Manila (MNL)", 
        selected_dest="Singapore (SIN)",
        selected_season="Standard",
        fare=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "<h2>AI Core offline. Run train.py first.</h2>"

    airline = request.form.get("airline")
    source = request.form.get("source")
    destination = request.form.get("destination")
    season = request.form.get("season")

    # 1. Distance Calculation
    coord1 = ASIAN_GEO_DATABASE[source]
    coord2 = ASIAN_GEO_DATABASE[destination]
    distance_km = geodesic(coord1, coord2).km

    # 2. AI Model Base Prediction
    input_df = pd.DataFrame([{"distance_km": distance_km}])
    base_prediction = model.predict(input_df)[0]

    # 3. Apply Multipliers & Analytics
    is_premium = "airlines" in airline.lower() or "singapore" in airline.lower() or "japan" in airline.lower()
    tier_multiplier = 1.30 if is_premium else 1.00
    
    season_multiplier = 1.00
    if season == "Peak":
        season_multiplier = 1.25
    elif season == "OffPeak":
        season_multiplier = 0.85
        
    final_fare = base_prediction * tier_multiplier * season_multiplier
    
    # Calculate extra costs added by choices
    surge_fees = final_fare - base_prediction
    if surge_fees < 0:
        surge_fees = 0

    # 4. Flight Duration & Layouts
    total_hours = distance_km / 850.0
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)
    formatted_duration = f"{hours}h {minutes}m"
    needs_layover = distance_km > 3000

    # 5. Dynamic Smart Savings Advice Logic
    if season == "Peak":
        advice = "Switching your scheduling to Standard or Off-Peak season could instantly shave up to 25% off this route target."
    elif is_premium:
        advice = "Cebu Pacific or AirAsia operate budget options for this sector that eliminate premium tier fleet fees."
    else:
        advice = "Smart Choice! You are optimizing your budget with a baseline carrier rate layout."

    return render_template(
        "index.html", 
        fare=f"{round(float(final_fare), 2):,}", 
        distance=f"{round(distance_km, 2):,}",
        duration=formatted_duration,
        needs_layover=needs_layover,
        base_fare=f"{round(float(base_prediction), 2):,}",
        surge=f"{round(float(surge_fees), 2):,}",
        advice=advice,
        selected_airline=airline,
        selected_source=source,
        selected_dest=destination,
        selected_season=season
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)