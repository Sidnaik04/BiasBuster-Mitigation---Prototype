import sys
import joblib

class FairModelBundle:
    pass

sys.modules['__main__'].FairModelBundle = FairModelBundle
if '__mp_main__' in sys.modules:
    sys.modules['__mp_main__'].FairModelBundle = FairModelBundle

path = "/home/sidnaik04/Documents/BiasBuster/mitigation_prototype/app/artifacts/models/5b76bb93b0744824a8eb94e6bf19d767_best_fair_model.pkl"

try:
    model = joblib.load(path)
    print("Loaded model successfully!")
    print("Model attributes:", dir(model))
    for k in dir(model):
        if not k.startswith("__"):
            print(f"{k}: {type(getattr(model, k))}")
except Exception as e:
    print("Error:", e)
