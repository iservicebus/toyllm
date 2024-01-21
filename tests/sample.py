import spacy
nlp =  spacy.load("en_core_web_sm")
with open("./tiny.txt", "r") as file:
    text = file.read()          

doc = nlp(text)

word_counts = doc.count_by(spacy.attrs.ORTH)

most_frequent_word = max(word_counts, key=word_counts.get)
most_frequent_word_token = word_counts.get(most_frequent_word)



print(f" most frequency word is {most_frequent_word_token}")
