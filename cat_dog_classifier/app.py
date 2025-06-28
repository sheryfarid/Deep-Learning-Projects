import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNN  # your trained CNN class

# Load the model
device = torch.device("cpu")
model = CNN()
model.load_state_dict(torch.load("cat_dog_other.pth", map_location=device))
model.eval()

# Class mapping
class_names = ["Cat 🐱", "Dog 🐶", "Other ❌"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# UI
st.title("🐶🐱 Cat vs Dog Classifier (with Other ❌ detection)")
st.write("Upload an image and the model will predict whether it's a **Cat**, **Dog**, or **Other**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save uploaded image to 'uploads/' folder
    import os
    os.makedirs("uploads", exist_ok=True)
    image.save(f"uploads/{uploaded_file.name}")

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 128, 128]

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        confidence = confidence.item()
        pred_class = pred_class.item()

        # Threshold for unknown
        if pred_class == 2 or confidence < 0.75:
            st.markdown("### Prediction: ❌ **Does not match (Neither Cat nor Dog)**")
        else:
            st.markdown(f"### Prediction: **{class_names[pred_class]}**")
            st.markdown(f"**Confidence:** {confidence:.2%}")
