# ===== STEP 1: WARNINGS IGNORE =====
import warnings
warnings.filterwarnings("ignore")

# ===== STEP 2: IMPORTS =====
from flask import Flask, Blueprint, request, render_template
import pickle
import math

# ===== STEP 3: CREATE FLASK APP =====
app = Flask(__name__)

# ===== STEP 4: BLUEPRINT =====
crime_bp = Blueprint('crime', __name__)

# ===== STEP 5: LOAD MODEL =====
model = pickle.load(open('model.pkl', 'rb'))

# ===== HOTFIX =====
if not hasattr(model, "monotonic_cst"):
    model.monotonic_cst = None

print("Crime Prediction Model Loaded Successfully")

# ===== HOME PAGE =====
@crime_bp.route('/')
def home():
    return render_template("home.html")

# ===== PREDICTION PAGE =====
@crime_bp.route('/prediction')
def prediction():
    return render_template("prediction.html")

# ===== PREDICT ROUTE =====
@crime_bp.route('/predict', methods=['POST'])
def predict_result():

    # ===== CITY MAPPING =====
    city_names = {
        '0': 'Ahmedabad',
        '1': 'Bengaluru',
        '2': 'Chennai',
        '3': 'Coimbatore',
        '4': 'Delhi',
        '5': 'Ghaziabad',
        '6': 'Hyderabad',
        '7': 'Indore',
        '8': 'Jaipur',
        '9': 'Kanpur',
        '10': 'Kochi',
        '11': 'Kolkata',
        '12': 'Kozhikode',
        '13': 'Lucknow',
        '14': 'Mumbai',
        '15': 'Nagpur',
        '16': 'Patna',
        '17': 'Pune',
        '18': 'Surat'
    }

    # ===== CRIME TYPES =====
    crimes_names = {
        '0': 'Crime Committed by Juveniles',
        '1': 'Crime against SC',
        '2': 'Crime against ST',
        '3': 'Crime against Senior Citizen',
        '4': 'Crime against children',
        '5': 'Crime against women',
        '6': 'Cyber Crimes',
        '7': 'Economic Offences',
        '8': 'Kidnapping',
        '9': 'Murder'
    }

    # ===== POPULATION =====
    population = {
        '0': 63.50,
        '1': 85.00,
        '2': 87.00,
        '3': 21.50,
        '4': 163.10,
        '5': 23.60,
        '6': 77.50,
        '7': 21.70,
        '8': 30.70,
        '9': 29.20,
        '10': 21.20,
        '11': 141.10,
        '12': 20.30,
        '13': 29.00,
        '14': 184.10,
        '15': 25.00,
        '16': 20.50,
        '17': 50.50,
        '18': 45.80
    }

    # ===== GET FORM DATA =====
    city_code = request.form["city"]
    crime_code = int(request.form["crime"])
    year = int(request.form["year"])

    pop = population[city_code]

    # ===== POPULATION ADJUSTMENT =====
    year_diff = year - 2011
    pop = pop + (0.01 * year_diff * pop)

    # ===== ONE HOT ENCODING =====
    city_encoded = [0] * 18
    city_index = int(city_code)

    if city_index < 18:
        city_encoded[city_index] = 1

    # ===== MODEL INPUT =====
    input_features = [year, pop, crime_code] + city_encoded

    # ===== PREDICTION =====
    crime_rate = model.predict([input_features])[0]

    print("Predicted Crime Rate:", crime_rate)

    city_name = city_names[city_code]
    crime_type = crimes_names[str(crime_code)]

    # ===== CRIME STATUS =====
    if crime_rate <= 150:
        crime_status = "Very Low Crime Area"
    elif crime_rate <= 200:
        crime_status = "Low Crime Area"
    elif crime_rate <= 250:
        crime_status = "High Crime Area"
    else:
        crime_status = "Very High Crime Area"

    cases = math.ceil(crime_rate * pop)

    return render_template(
        'result.html',
        city_name=city_name,
        crime_type=crime_type,
        year=year,
        crime_status=crime_status,
        crime_rate=round(crime_rate, 2),
        cases=cases,
        population=round(pop, 2)
    )

# ===== REGISTER BLUEPRINT =====
app.register_blueprint(crime_bp)

# ===== RUN APP =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
