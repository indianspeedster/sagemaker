import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import cuda


device = 'cuda' if cuda.is_available() else 'cpu'

def model_fn(model_dir):
  # Load model from HuggingFace Hub
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    return model, tokenizer

def predict_fn(data, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    
    # Tokenize sentences
    sentences = data.pop("inputs", data)
    input_ids = tokenizer.encode(sentences, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = input_ids.to(device)

    # Compute token embeddings
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # return dictonary, which will be json serializable
    return {"summary": output_text}
