import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os


model = load_model('waste_sorter_mobilenetv2.h5')

with open('class_names.json', 'r') as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}


def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img_array = np.expand_dims(img, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_class = index_to_class[np.argmax(prediction)]

        
        display_img = cv2.imread(image_path)
        cv2.putText(display_img, f"Prediction: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"{os.path.basename(image_path)} --> {predicted_class}")
    except Exception as e:
        print(f"Error with {image_path}: {e}")


image_folder = "C:\\Users\\Admin\\Documents\\VIT\\Sem 2\\Python Lab -Atharva\\Python-CP\\Final CP\\test"  
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        full_path = os.path.join(image_folder, filename)
        predict_image(full_path)
