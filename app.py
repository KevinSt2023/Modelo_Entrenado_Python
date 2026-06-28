from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ------------------------------------------------------------------
# Cargar el modelo, el scaler y el orden exacto de columnas usado
# durante el entrenamiento (generados en el notebook de Jupyter)
# ------------------------------------------------------------------
modelo = joblib.load("modelo_rf_canal_venta.pkl")
scaler = joblib.load("scaler_canal_venta.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def construir_fila(form):
    """
    Convierte los datos del formulario en una fila con EXACTAMENTE
    las mismas columnas (y en el mismo orden) que usó el modelo al
    entrenarse: variables numéricas + dummies de Brand/Shoe_Type/Color/Country.
    """
    datos = {
        "Price_USD": float(form["Price_USD"]),
        "Units_Sold": float(form["Units_Sold"]),
        "Revenue_USD": float(form["Revenue_USD"]),
        "Brand": form["Brand"],
        "Shoe_Type": form["Shoe_Type"],
        "Color": form["Color"],
        "Country": form["Country"],
    }

    fila = pd.DataFrame([datos])

    # Mismo get_dummies que en el notebook
    fila = pd.get_dummies(fila, columns=["Brand", "Shoe_Type", "Color", "Country"])

    # Alinear columnas con las del entrenamiento:
    # - si falta una columna dummy (porque esa categoría fue la "base"
    #   o no se generó), se rellena con 0
    # - se descarta cualquier columna que el modelo no conozca
    fila = fila.reindex(columns=feature_columns, fill_value=0)

    return fila


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    valores = {
        "Price_USD": "",
        "Units_Sold": "",
        "Revenue_USD": "",
    }

    if request.method == "POST":
        valores["Price_USD"] = request.form["Price_USD"]
        valores["Units_Sold"] = request.form["Units_Sold"]
        valores["Revenue_USD"] = request.form["Revenue_USD"]

        fila = construir_fila(request.form)
        fila_escalada = scaler.transform(fila)

        prediccion = modelo.predict(fila_escalada)[0]
        probabilidad = modelo.predict_proba(fila_escalada)[0][1]  # prob. de ser Online

        if prediccion == 1:
            resultado = f"✅ Predicción: Canal Online (probabilidad {probabilidad:.0%})"
        else:
            resultado = f"🏬 Predicción: No Online — Mall/Retail Store (probabilidad Online {probabilidad:.0%})"

    return render_template("index.html", resultado=resultado, valores=valores)


if __name__ == "__main__":
    app.run(debug=True)
