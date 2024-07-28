import gardio as gr 
import streamlit as st
import AutoTokenizer, AutoAdapterModel

# Load the base model
base_model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your base model name
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoAdapterModel.from_pretrained(base_model_name)

# Load the adapter configuration and model files
adapter_config_path = "C:/Users/Administrator/Desktop/Ayn_Rand/adapter_config.json"  # Replace with your adapter config path
adapter_model_path = "C:/Users/Administrator/Desktop/Ayn_Rand/adapter_model.safetensors"  # Replace with your adapter model path

# Load the adapter into the model
adapter_name = "custom_adapter"  # Define your adapter name
model.load_adapter(adapter_config_path, model_file=adapter_model_path, load_as=adapter_name)

# Activate the adapter
model.set_active_adapters(adapter_name)

st.title("🤖 Chatbot with Adapter-Enhanced Model")
st.write("Interact with your custom adapter-enhanced model. Type a message and get responses!")

# Initialize or retrieve the chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize Gardio
chatbot = Gardio(model=model, tokenizer=tokenizer)

# Define responses for greetings
@chatbot.on_event("welcome")
def welcome_handler(payload):
    return "Welcome! Type a message and get responses from the chatbot."

# Define responses for user messages
@chatbot.on_message
def message_handler(payload):
    user_input = payload["message"]
    response = chatbot.generate_response(user_input)
    return response

# Run Gardio
if __name__ == "__main__":
    chatbot.run()
