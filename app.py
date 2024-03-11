import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Load model and tokenizer
checkpoint = "bigcode/starcoder2-15b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, verify=False)

# Initialize history and context variables
history = []
context = ""

# Define Streamlit app
def main():
    global history, context
    
    st.title("Streamlit App for StarCoder Model")
    
    # Show previous context
    st.sidebar.title("Previous Context")
    st.sidebar.text_area("Context:", value=context, height=200)

    # User input
    user_input = st.text_input("Enter your code snippet:", "")

    if st.button("Submit"):
        # Store user input in history and context
        history.append(user_input)
        context = "\n".join(history[-5:])  # Retain last 5 inputs
        
        # Run StarCoder model
        inputs = tokenizer.encode(user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs)
        generated_code = tokenizer.decode(outputs[0])
        
        # Show generated code
        st.write("Generated Code:")
        st.code(generated_code)

if __name__ == "__main__":
    main()
