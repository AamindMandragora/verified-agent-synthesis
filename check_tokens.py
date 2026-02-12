from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokens = ["<<", ">>", "<", ">"]
for t in tokens:
    ids = tokenizer.encode(t, add_special_tokens=False)
    print(f"Token: {t}, IDs: {ids}, Decoded: {tokenizer.decode(ids)}")
