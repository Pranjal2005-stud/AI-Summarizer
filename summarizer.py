from huggingface_hub import login


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
MODEL_NAME = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def split_text(text, chunk_size=800):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def summarize_chunk(chunk):

    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        forced_bos_token_id=tokenizer.bos_token_id
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_summary(text):

    # Images often contain very little text. 
    # If the text is already short, summarization will fail or hallucinate due to min_length constraints.
    if len(text.split()) < 40:
        return text

    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunk_summaries = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=130,
            min_length=30,
            num_beams=4
        )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        chunk_summaries.append(summary)

    combined_summary = " ".join(chunk_summaries)

    inputs = tokenizer(
        combined_summary,
        return_tensors="pt",
        truncation=True
    )

    final_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4
    )

    final_summary = tokenizer.decode(
        final_ids[0],
        skip_special_tokens=True
    )

    return final_summary