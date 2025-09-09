import streamlit as st
from create_mem_2 import analyze_colors, get_clip_embedding, run_query

st.set_page_config(page_title="Fashion Stylist AI", page_icon="👗", layout="centered")
st.title(" Fashion Police")

uploaded = st.file_uploader("📸 Upload your outfit image", type=["jpg", "png"])

outfit_info = ""  # to store extracted details

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())

    st.image("temp.jpg", caption="Your Uploaded Outfit", use_container_width=True)

    st.subheader("🎨 Outfit Color Analysis")
    color_feedback, named_colors = analyze_colors("temp.jpg")
    outfit_info = f"Detected outfit colors: {', '.join(named_colors)}"
    st.markdown(color_feedback)

    clip_vec = get_clip_embedding("temp.jpg")
    st.success("✔ Image embedding created for outfit similarity checks.")

st.subheader("💬 Ask the Stylist")
user_query = st.text_input("Ask about anything fashion related( or upload your outfit and ask about it)")

if user_query:
    response = run_query(user_query, outfit_info=outfit_info)
    st.subheader("🧾 Fashion Q&A Result")
    st.markdown(response)

