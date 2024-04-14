from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app = FastAPI(
    title = "Deploy Glaucoma detection",
    version = "0.0.1"
    )


# ----------------------------------------
# LOAD MODEL
#-----------------------------------------
model = joblib.load("model/logistic_regression_model_v01.pkl")

@app.post("/api/v1/predict_glaucoma", tags=["Glaucoma"])
async def predict(
    Age: float,
    Intraocular_Pressure_IOP: float,
    Cup_to_Disc_Ratio_CDR: float,
    Hypertension: float,
    Diabetes: float,
    Primary_Open_Angle_Glaucoma: float,
    Juvenile_Glaucoma: float,
    Congenital_Glaucoma: float,
    Normal_Tension_Glaucoma: float,
    Angle_Closure_Glaucoma: float,
    Secondary_Glaucoma: float,
    Redness_in_the_eye: float,
    Nausea: float,
    Tunnel_vision: float,
    Vision_loss: float,
    Blurred_vision: float,
    Vomiting: float,
    Eye_pain: float,
    Halos_around_lights: float,
    Aspirin: float,
    Ibuprofen: float,
    Omeprazole: float,
    Amoxicillin: float,
    Lisinopril: float,
    Metformin: float,
    Atorvastatin: float,
    Sensitivity: float,
    Specificity: float,
    RNFL_Thickness: float,
    GCC_Thickness: float,
    Retinal_Volume: float,
    Macular_Thickness: float
):

    dictionary = {
        'Age': Age,
        'Intraocular Pressure (IOP)': IntraocularPressure,
        'Cup-to-Disc Ratio (CDR)': Cup_to_DiscRatio,
        'Pachymetry': Pachymetry,
        'Hypertension': Hypertension,
        'Diabetes': Diabetes,
        'Primary Open-Angle Glaucoma': Primary_Open_Angle_Glaucoma,
        'Juvenile Glaucoma': Juvenile_Glaucoma,
        'Congenital Glaucoma': Congenital_Glaucoma,
        'Normal-Tension Glaucoma': Normal_Tension_Glaucoma,
        'Angle-Closure Glaucoma': Angle_Closure_Glaucoma,
        'Secondary Glaucoma': Secondary_Glaucoma,
        'Redness in the eye': Redness_in_the_eye,
        'Nausea': Nausea,
        'Tunnel vision': Tunnel_vision,
        'Vision loss': Vision_loss,
        'Blurred vision': Blurred_vision,
        'Vomiting': Vomiting,
        'Eye pain': Eye_pain,
        'Halos around lights': Halos_around_lights,
        'Aspirin': Aspirin,
        'Ibuprofen': Ibuprofen,
        'Omeprazole': Omeprazole,
        'Amoxicillin': Amoxicillin,
        'Lisinopril': Lisinopril,
        'Metformin': Metformin,
        'Atorvastatin': Atorvastatin,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity,
        'RNFL Thickness': RNFL_Thickness,
        'GCC Thickness': GCC_Thickness,
        'Retinal Volume': Retinal_Volume,
        'Macular Thickness': Macular_Thickness
    }

    try:
        df = pd.DataFrame(dictionary, index= [0])
        prediction = model.predict(df)
        return JSONResponse(
            status_code= status.HTTP_200_ok,
            content=prediction[0]
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code = status.HTTP_400_BAD_REQUEST
        )

print()


