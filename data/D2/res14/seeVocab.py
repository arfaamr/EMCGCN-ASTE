"""
author: arfaa rashid
date: nov 17, 2023
purpose: try to see what the data looks like (.vocab files specifically), to convert LADy's data to be compatible with emc
"""


import pickle

class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        print("INITING")
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]
        
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)    # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)


    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def see_vocab(vocab_path: str):
        f = open(vocab_path, "rb")          #rb is read binary
        content = f.read()
        print(content)
        
post_vocab = VocabHelp.load_vocab('vocab_post.vocab')       #modified path bc my code file is in data folder for convenience

#print(post_vocab[0])       #? this works for digital ocean tutorial but not for us :( maybe the object they unpickle is subscriptable, and ours isnt
VocabHelp.see_vocab('vocab_post.vocab')


print(type(post_vocab))
print("Goodnight!")

#note: because it's a static method, we don't init. so the only thing happening is opening, loading.

