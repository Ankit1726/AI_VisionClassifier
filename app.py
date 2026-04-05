import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# -----------------------------
# 🎨 PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Vision Classifier",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# 🎨 MODERN DARK UI
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}

/* Title Gradient */
.title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(to right, #8b5cf6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Card UI */
.card {
    background: rgba(30, 41, 59, 0.6);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border: 2px dashed #6366f1;
    border-radius: 12px;
    padding: 20px;
    background-color: rgba(30, 41, 59, 0.6);
}

/* Prediction box */
.pred-box {
    background: rgba(15, 23, 42, 0.8);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧠 MODEL (UNCHANGED)
# -----------------------------
class CNN(nn.Module): 
    def __init__(self):
        super(CNN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,x): 
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        return self.fc_layers(x)

# -----------------------------
# 🔁 LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cifar10_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 📊 CLASSES
# -----------------------------
classes = [
    "Airplane ✈️", "Automobile 🚗", "Bird 🐦", "Cat 🐱", "Deer 🦌",
    "Dog 🐶", "Frog 🐸", "Horse 🐴", "Ship 🚢", "Truck 🚚"
]

# -----------------------------
# 🖼️ TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# -----------------------------
# 🎛️ SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Control Panel")
top_k = st.sidebar.slider("Top Predictions", 1, 5, 3)

st.sidebar.markdown("### 👨‍💻 About")
st.sidebar.write("CIFAR-10 CNN Classifier")

# -----------------------------
# 🌟 HEADER
# -----------------------------
st.markdown('<div class="title">🧠 AI Vision Classifier</div>', unsafe_allow_html=True)
st.write("Upload an image and get predictions instantly.")

# -----------------------------
# 📂 UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# -----------------------------
# 🔍 PREDICTION
# -----------------------------
if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, width="stretch")  # ✅ FIXED HERE
        st.markdown('</div>', unsafe_allow_html=True)

    img = transform(image).unsqueeze(0)

    with st.spinner("Analyzing..."):
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, top_k)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Predictions")

        for i in range(top_k):
            label = classes[top_idxs[i]]
            confidence = float(top_probs[i])

            st.markdown(f'<div class="pred-box"><b>{label}</b></div>', unsafe_allow_html=True)
            st.progress(confidence)
            st.write(f"{confidence*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

    st.success(f"✅ Final Prediction: {classes[top_idxs[0]]}")

# -----------------------------
# 📌 MODEL INFO
# -----------------------------
st.markdown("---")
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Model Overview")
st.write("""
- Dataset: CIFAR-10  
- CNN (3 Conv Layers)  
- Input: 32x32  
- Output: 10 Classes  
""")

st.markdown('</div>', unsafe_allow_html=True)