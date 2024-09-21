#installing the required libraries
!pip install -q gradio
!pip install -q git+https://github.com/huggingface/transformers.git

#importing the libraries
import gradio as gr
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

#tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#generating the text based on the input prompt
def generate_text(inp):
  input_ids = tokenizer.encode(inp, return_tensors='tf')
  beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
  output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return ".".join(output.split(".")[:-1]) + "."

#gradio interface design
gr.Interface(generate_text,'textbox','textbox',title = 'GPT-2', description = 'text generation model').launch()
