import streamlit as st
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, Tool
from twilio.rest import Client
from typing import TypedDict

# Load environment variables
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# WhatsApp Tool
class WhatsAppMessageInput(TypedDict):
    to: str
    message: str

class WhatsAppTool(Tool):
    name = "send_whatsapp_message"
    description = "Send a WhatsApp message using Twilio"
    input_type = WhatsAppMessageInput

    def __init__(self):
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    async def call(self, input: WhatsAppMessageInput) -> str:
        try:
            message = self.client.messages.create(
                body=input["message"],
                from_=TWILIO_WHATSAPP_NUMBER,
                to=f"whatsapp:{input['to']}"
            )
            return f"✅ WhatsApp message sent to {input['to']}!"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Model & Agent Setup
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)
model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
whatsapp_tool = WhatsAppTool()

agent = Agent(
    name="Matchmaker Auntie",
    instructions="You are a matchmaking assistant. Collect user details and help connect them on WhatsApp.",
    model=model,
    tools=[whatsapp_tool],
)

# 🌸 Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        color: #000;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.15);
    }
    h1, h2 {
        color: #c71585;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #da70d6, #ba55d3);
        color: white;
        font-weight: bold;
        border-radius: 30px;
        font-size: 16px;
        border: none;
        padding: 0.6rem 1.5rem;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ba55d3, #9932cc);
        transform: scale(1.05);
    }
    .stTextInput > div > input,
    .stTextArea > div > textarea {
        background-color: #fff0f5;
        border: 1px solid #dda0dd;
        border-radius: 12px;
        padding: 0.75rem;
    }
    .stRadio > div {
        background-color: #ffe4f1;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #dda0dd;
    }
    </style>
""", unsafe_allow_html=True)

# Banner (optional)
st.image("banner.png", use_container_width=True)

st.title("💞 Rishta Wali Auntie")
st.markdown("### 🧕 *Find Your Perfect Match — The AI Way!*")
st.markdown("📲 Submit your rishta profile and Auntie will get you connected on WhatsApp.")

st.markdown("---")
st.markdown("""<h2 style='text-align:center;'>💘 Matchmaking Form</h2>""", unsafe_allow_html=True)

# 📋 Match Form
with st.form("match_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("🧑‍💼 Full Name", placeholder="e.g. Ali Khan")
        age = st.slider("🎂 Age", 18, 80, 25)
        gender = st.radio("⚧️ Gender", ["Male", "Female", "Other"])
    with col2:
        profession = st.text_input("💼 Profession", placeholder="e.g. Doctor, Engineer")
        phone = st.text_input("📱 WhatsApp Number", placeholder="+92xxxxxxxxxx")
        about = st.text_area("📝 About You", placeholder="What kind of partner are you looking for?")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("💌 Submit & Get Matched")

# 🚀 Submission Handling
if submitted:
    user_prompt = f"""
    A new matchmaking request:

    Name: {name}
    Age: {age}
    Gender: {gender}
    Profession: {profession}
    Phone: {phone}
    About: {about}
    """

    async def run_agent():
        st.info("🧠 Auntie is thinking... Please wait!")
        result = Runner.run_streamed(agent, user_prompt)

        full_response = ""
        response_placeholder = st.empty()

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                full_response += event.data.delta
                response_placeholder.markdown(f"💬 {full_response}")

        # ✅ Confirmation to User
        st.info("📨 Sending confirmation to you...")
        user_msg = f"Hello {name}, Rishta Wali Auntie received your profile! 💖 We'll contact you soon."
        await whatsapp_tool.call({
            "to": phone,
            "message": user_msg
        })

        # ✅ Notification to Admin
        admin_number = "YOUR_ADMIN_WHATSAPP_NUMBER"  # e.g. +923001234567
        admin_msg = f"📥 New Rishta Submission!\n\nName: {name}\nAge: {age}\nGender: {gender}\nProfession: {profession}\nPhone: {phone}\nAbout: {about}"
        await whatsapp_tool.call({
            "to": admin_number,
            "message": admin_msg
        })

        st.success("✅ Submission complete. WhatsApp messages sent!")

    asyncio.run(run_agent())
