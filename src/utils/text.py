def text_to_char_sequence(text):
    text = text.lower().strip()
    return " ".join(list("|".join(text.split())))
