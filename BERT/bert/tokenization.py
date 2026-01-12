def wordpiece_tokenizer(text, vocab, unk_token='[UNK]'):
    tokens = []
    start = 0

    while start < len(text):
        end = len(text)
        found = False

        while end > start:
            substr = text[start:end]
            if start > 0:
                substr = "##" + substr

            if substr in vocab:
                tokens.append(substr)
                start = end
                found = True
                break
            end -= 1

        if not found:
            return [unk_token]

    return tokens
