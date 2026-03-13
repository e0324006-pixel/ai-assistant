def format_dataset(example):
    # Handle different dataset formats
    if "text" in example:
        text = example.get("text", "")
    elif "instruction" in example and "output" in example:
        instruction = example.get("instruction", "")
        output = example.get("output", "")
        text = instruction + " " + output
    elif "prompt" in example:
        text = example.get("prompt", "")
    elif "chosen" in example:
        # For preference datasets
        text = example.get("chosen", "")
    else:
        # Fallback: combine all string values
        text = " ".join([str(v) for v in example.values() if isinstance(v, str)])
    
    return {"text": text}