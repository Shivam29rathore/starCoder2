import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Install and import requirements
import subprocess

# Create requirements.txt
with open("requirements.txt", "w") as f:
    f.write("transformers==4.11.3\n")
    f.write("torch==1.9.0\n")
    f.write("streamlit==1.1.0\n")

# Install requirements
subprocess.call(["pip", "install", "-r", "requirements.txt"])

# Load model and tokenizer
checkpoint = "bigcode/starcoder2-15b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, verify=False)

# Define Streamlit app
def main():
    st.title("Streamlit App for StarCoder Model")

    # Run StarCoder model
    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    st.write("Generated Code:")
    st.code(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
