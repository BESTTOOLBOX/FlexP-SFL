import os
import pickle
from train_plain import train_plain
from train_AB import train_ab
from evaluate import evaluate

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model.state_dict(), f)
    print(f"Model saved to {path}")

#print("=== Running Plain GPT2 Fine-tune ===")
#model_plain = train_plain()
#evaluate(model_plain)
#save_model(model_plain, "models/model_plain.pkl")

print("\n=== Running A-B Matrix GPT2 ===")
model_ab = train_ab()
save_model(model_ab, "models/model_ab.pkl")