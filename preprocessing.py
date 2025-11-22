import gensim
import nltk
import numpy as np
from nltk.corpus import wordnet

# column_names = ['class', 'text']
# text_data = pd.read_csv("data/ecommerceDataset.csv", name=column_names, heading=None)


def label_enc(text_data):
    labels_mapping = {
        "Household": 0,
        "Electronics": 1,
        "Books": 2,
        "Clothing & Accessories": 3,
    }
    text_data["class"] = text_data["class"].map(labels_mapping)
    return text_data


# nltk pos -> worned pos tags mapping
def nltk_pos_to_wordnet_pos(nltk_pos):
    if nltk_pos.startswith("J"):
        return wordnet.ADJ
    elif nltk_pos.startswith("V"):
        return wordnet.VERB
    elif nltk_pos.startswith("N"):
        return wordnet.NOUN
    elif nltk_pos.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_text(text_data):
    all_texts = text_data["text"].fillna("").astype(str)
    stopwords = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.WordNetLemmatizer()
    formatted_texts = []
    for text in all_texts:
        text = text.lower()
        tokenized_text = nltk.word_tokenize(text)
        tokenized_text = [
            t for t in tokenized_text if t.isalpha() and t not in stopwords
        ]
        pos_tags = nltk.pos_tag(tokenized_text)  # nltk pos tags
        lemmatized_tokens = [
            lemmatizer.lemmatize(token, nltk_pos_to_wordnet_pos(pos))
            for token, pos in pos_tags
        ]
        formatted_texts.append(lemmatized_tokens)

    text_data["tokenized"] = formatted_texts
    return text_data


def get_embeddings(text_data):
    texts = text_data["tokenized"].tolist()
    model = gensim.models.Word2Vec(texts, vector_size=150, window=5, min_count=2, sg=0)

    embedding_dim = model.wv.vector_size

    embeddings = []
    for tokens in texts:
        word_vectors = []
        for word in tokens:
            if word in model.wv:
                word_vectors.append(model.wv[word])

        if word_vectors:
            sentence_emb = np.array(word_vectors, dtype=np.float32)
        else:
            sentence_emb = np.zeros((1, embedding_dim), dtype=np.float32)

        embeddings.append(sentence_emb)

    text_data["embeddings"] = embeddings
    return text_data, embedding_dim
