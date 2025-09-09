import matplotlib.colors as mcolors
from colorthief import ColorThief
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# ---------------------------
# Color Utilities (stable via matplotlib)
# ---------------------------
def closest_color(requested_color):
    min_colors = {}
    for name, hex_val in mcolors.CSS4_COLORS.items():
        r_c, g_c, b_c = mcolors.to_rgb(hex_val)
        r_c, g_c, b_c = [int(x*255) for x in (r_c, g_c, b_c)]
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_palette(image_path):
    """Extract color palette from an outfit image."""
    ct = ColorThief(image_path)
    return ct.get_palette(color_count=5)

def analyze_colors(file_path):
    """Suggest outfit improvements based on detected colors."""
    palette = get_palette(file_path)
    named_colors = [closest_color(c) for c in palette]
    base_color = named_colors[0]

    suggestions = f"🎨 Detected main colors: {', '.join(named_colors)}.\n"
    suggestions += f"👉 Your dominant shade is **{base_color}**.\n"

    # Fashion color theory
    if base_color in ["brown", "saddlebrown", "peru", "chocolate"]:
        suggestions += "✅ Brown pairs well with navy, cream, beige, and forest green.\n"
    elif base_color in ["black", "dimgray", "gray"]:
        suggestions += "✅ Black/Gray go with almost anything; add bold colors like red or cobalt for contrast.\n"
    elif base_color in ["white", "linen", "ivory"]:
        suggestions += "✅ White looks crisp with navy, olive, black, or pastels.\n"
    else:
        suggestions += "✨ Try complementary or contrasting accents to make the outfit pop.\n"

    return suggestions, named_colors


# ---------------------------
# CLIP Embeddings for Images
# ---------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def get_clip_embedding(image_path):
    """Generate image embeddings using CLIP."""
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        return clip_model.get_image_features(**inputs).cpu().numpy()


# ---------------------------
# LLM + Knowledge Base
# ---------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = OllamaEmbeddings(model="all-minilm")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

CUSTOM_PROMPT_TEMPLATE = """
You are a professional fashion stylist. 
Use the following information to answer the user's question with detailed style advice.

Context (fashion knowledge base): {context}
Outfit Info (from uploaded image): {outfit_info}
Question: {question}

Give a stylist-quality response with clear reasoning.
"""

def run_query(query: str, outfit_info: str = ""):
    # 1. Retrieve context from FAISS
    retrieved_docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Build final prompt
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "outfit_info", "question"]
    )

    # 3. LLM Chain
    llm_chain = LLMChain(llm=Ollama(model="mistral"), prompt=prompt)

    # 4. Run query with outfit info
    response = llm_chain.invoke({
        "context": context,
        "outfit_info": outfit_info,
        "question": query
    })

    return response["text"]
