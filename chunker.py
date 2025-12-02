def chunk_text(text, chunk_size=400):
    words = text.split()
    #Splits the text into words on whitespace — creates a list of words.
    #(list comprehension)
    #chunks by slicing the words list every chunk_size words. 
    # For each slice, it rejoins words with spaces into a chunk string.
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks

