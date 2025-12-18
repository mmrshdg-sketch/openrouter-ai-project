import os
import json
import uuid
import base64
import requests
import gradio as gr

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
CHAT_FILE = "chats.json"
MAX_MESSAGES = 1000

# ---------------- MODELS ----------------
MODELS = {
    "DeepSeek V3.1 Free": "nex-agi/deepseek-v3.1-nex-n1:free",
    "Mistral Devstral 2 (Coding)": "mistralai/devstral-2512:free",
    "KAT Coder Pro (Agentic Coding)": "kwaipilot/kat-coder-pro:free",
    "DeepSeek R1T2 Chimera": "tngtech/deepseek-r1t2-chimera:free",
    "Olmo 32B Think": "allenai/olmo-3.1-32b-think:free",
    "GLM 4.5 Air": "z-ai/glm-4.5-air:free",
    "Gemma 3 4B Vision": "google/gemma-3-4b-it:free",
    "Gemma 3 27B Vision": "google/gemma-3-27b-it:free",
    "Amazon Nova 2 Lite (Video)": "amazon/nova-2-lite-v1:free",
    "NVIDIA Nemotron VL (Video)": "nvidia/nemotron-nano-12b-v2-vl:free",
    "LLaMA 3.3 70B": "meta-llama/llama-3.3-70b-instruct:free",
    "GPT-OSS 20B": "openai/gpt-oss-20b:free",
    "Qwen3 Coder": "qwen/qwen3-coder:free"
}

# ---------------- CHAT MEMORY ----------------
def load_chats():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHAT_FILE, "w") as f:
        json.dump(chats, f, indent=2)

chats = load_chats()
if chats:
    current_chat_id = list(chats.keys())[0]
else:
    current_chat_id = str(uuid.uuid4())
    chats[current_chat_id] = []
    save_chats(chats)

def add_message(role, content):
    chats[current_chat_id].append({"role": role, "content": content})
    chats[current_chat_id] = chats[current_chat_id][-MAX_MESSAGES:]
    save_chats(chats)

# ---------------- HELPERS ----------------
def encode_file(file_path, mime):
    with open(file_path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def send_message(model, text, file, image, video):
    content = []
    if text: content.append({"type": "text", "text": text})
    if file:
        with open(file, "r", errors="ignore") as f:
            content.append({"type": "text", "text": f"[FILE]\n" + f.read()[:12000]})
    if image: content.append({"type": "image_url", "imageUrl": {"url": encode_file(image, "image/png")}})
    if video: content.append({"type": "video_url", "videoUrl": {"url": encode_file(video, "video/mp4")}})

    # Add user message
    add_message("user", content)

    # Send request
    payload = {"model": MODELS[model], "messages": chats[current_chat_id]}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload)
    response = r.json()["choices"][0]["message"]["content"]

    # Add AI response
    add_message("assistant", response)
    return response

def new_chat():
    global current_chat_id
    current_chat_id = str(uuid.uuid4())
    chats[current_chat_id] = []
    save_chats(chats)
    return ""

# ---------------- UI ----------------
with gr.Blocks(css="""
#bottom {position: sticky; bottom: 0; background: #0f0f14; padding: 8px; z-index: 99; display: flex; gap: 5px;}
#overlay {position: fixed; inset: 0; backdrop-filter: blur(8px); background: rgba(0,0,0,0.5); z-index: 999; display: none;}
#modal {background: #1e1e2e; border-radius: 16px; padding: 20px; width: 70%; margin: auto; margin-top: 10%;}
button {border-radius: 10px !important; background-color: #2e2e3e; color: white; border: none; padding: 5px 10px;}
button:hover {background-color: #4a90e2;}
textarea, input, select {background-color: #2e2e3e; color: white; border-radius: 8px; border: none;}
""") as demo:

    gr.Markdown("## 🧠 DeepSeek Chat with Memory + Overlay Tools")

    chat_display = gr.Textbox(lines=20, interactive=False)

    # Tool overlay
    with gr.Row(elem_id="overlay") as overlay:
        with gr.Column(elem_id="modal"):
            gr.Markdown("### Upload Tools")
            file_input = gr.File(label="📄 Upload File")
            img_input = gr.Image(label="🖼 Upload Image", type="filepath")
            vid_input = gr.File(label="🎥 Upload Video", file_types=[".mp4"])
            close_btn = gr.Button("Close Tools")

    # Bottom bar
    with gr.Row(elem_id="bottom"):
        plus_btn = gr.Button("➕")
        text_input = gr.Textbox(placeholder="Type your message...", lines=2)
        model_dropdown = gr.Dropdown(list(MODELS.keys()), value="DeepSeek V3.1 Free")
        send_btn = gr.Button("Send")
        new_chat_btn = gr.Button("New Chat")

    # Button logic
    plus_btn.click(lambda: gr.update(visible=True), None, overlay)
    close_btn.click(lambda: gr.update(visible=False), None, overlay)
    send_btn.click(send_message, [model_dropdown, text_input, file_input, img_input, vid_input], chat_display)
    new_chat_btn.click(new_chat, None, chat_display)

demo.launch()
