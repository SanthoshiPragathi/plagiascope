"""
Semantic Knapsack Plagiarism Engine
Word2Vec Embeddings + 0/1 Knapsack Optimization
"""
import re
import numpy as np
from collections import Counter

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","of","and","or","but",
    "for","with","as","by","from","this","that","these","those","was",
    "are","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","not","no",
    "so","if","then","than","just","also","its","i","we","you","he","she",
    "they","our","your","their","my","me","him","her","us","them","which",
    "who","whom","whose","what","when","where","why","how","all","each",
    "every","both","few","more","most","other","some","such","only","own",
    "same","too","very","s","t","now","ll","re","ve","don","didn","isn",
    "wasn","weren","hasn","haven","hadn","won","wouldn","couldn","shouldn"
}

SYNONYM_GROUPS = [
    {"doctor","physician","surgeon","medic","clinician"},
    {"medicine","medication","drug","remedy","treatment","therapy"},
    {"patient","sick","ill","unwell","person","individual"},
    {"fever","temperature","heat"},
    {"heal","recover","recuperate","improve","restore"},
    {"reduce","lower","decrease","minimize","lessen","diminish"},
    {"give","administer","provide","supply","offer","deliver"},
    {"help","assist","aid","support","enable","allow"},
    {"fast","quick","rapid","swift","speedy"},
    {"big","large","huge","enormous","massive","great"},
    {"small","tiny","little","minute","compact"},
    {"start","begin","initiate","commence","launch"},
    {"end","finish","complete","conclude","terminate"},
    {"make","create","build","construct","produce","generate"},
    {"use","utilize","employ","apply","implement"},
    {"show","display","demonstrate","present","reveal","exhibit"},
    {"learn","study","acquire","understand","grasp","comprehend"},
    {"teach","educate","instruct","train","guide"},
    {"increase","grow","rise","expand","boost","enhance"},
    {"decrease","shrink","fall","drop","decline","reduce"},
    {"important","significant","crucial","vital","key","critical"},
    {"new","novel","fresh","modern","recent","innovative"},
    {"good","excellent","great","superior","fine","positive"},
    {"bad","poor","inferior","negative","terrible","awful"},
    {"difficult","hard","challenging","complex","tough"},
    {"easy","simple","straightforward","effortless","basic"},
    {"find","discover","detect","identify","locate","uncover"},
    {"say","state","claim","assert","declare","mention"},
    {"think","believe","consider","assume","suppose","feel"},
    {"know","understand","realize","recognize","acknowledge"},
    {"data","information","dataset","corpus","records"},
    {"model","system","algorithm","method","approach","technique"},
    {"result","outcome","output","finding","conclusion"},
    {"document","text","paper","article","report","content"},
    {"similarity","resemblance","likeness","correspondence"},
    {"detect","identify","find","discover","recognize","spot"},
    {"accurate","precise","correct","exact","reliable"},
    {"efficient","effective","optimal","fast","quick"},
    {"convert","transform","translate","change","process"},
    {"subset","part","component","portion","segment","section"},
    {"enable","allow","permit","let","facilitate"},
    {"program","code","software","application","system"},
    {"compute","calculate","process","determine","evaluate"},
    {"glucose","sugar","food","energy","fuel","nutrient"},
    {"sunlight","light","sun","radiation","energy"},
    {"plant","organism","species","life","biological"},
]

WORD_TO_SYN_ID = {}
for gid, group in enumerate(SYNONYM_GROUPS):
    for word in group:
        WORD_TO_SYN_ID[word] = gid


class Word2Vec:
    def __init__(self, dim=50, window=3, epochs=300, lr=0.03):
        self.dim = dim; self.window = window
        self.epochs = epochs; self.lr = lr
        self.vocab = []; self.word2idx = {}
        self.syn0 = None; self.syn1 = None

    def _build_vocab(self, sentences):
        freq = Counter(w for s in sentences for w in s)
        self.vocab = [w for w, c in freq.items() if c >= 1]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.syn0 = np.random.uniform(-0.5/self.dim, 0.5/self.dim, (V, self.dim))
        self.syn1 = np.zeros((V, self.dim))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _train_pair(self, ci, ctx, lr):
        V = len(self.vocab)
        h = self.syn0[ci]
        out = self._sigmoid(h @ self.syn1[ctx])
        err = (out - 1.0) * lr
        self.syn1[ctx] -= err * h
        grad = err * self.syn1[ctx].copy()
        for neg in np.random.randint(0, V, 5):
            if neg == ctx: continue
            out_n = self._sigmoid(h @ self.syn1[neg])
            err_n = out_n * lr
            self.syn1[neg] -= err_n * h
            grad += err_n * self.syn1[neg]
        self.syn0[ci] -= grad

    def train(self, sentences):
        self._build_vocab(sentences)
        if len(self.vocab) < 3: return
        pairs = []
        for s in sentences:
            idxs = [self.word2idx[w] for w in s if w in self.word2idx]
            for i, c in enumerate(idxs):
                for j in range(max(0, i-self.window), min(len(idxs), i+self.window+1)):
                    if i != j: pairs.append((c, idxs[j]))
        if not pairs: return
        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            lr = self.lr * (1 - epoch/self.epochs) + 0.001
            for c, ctx in pairs:
                self._train_pair(c, ctx, lr)

    def get_vector(self, word):
        if word not in self.word2idx: return None
        return self.syn0[self.word2idx[word]]

    def ngram_vector(self, ngram):
        vecs = [self.get_vector(w) for w in ngram if self.get_vector(w) is not None]
        if not vecs: return None
        return np.mean(vecs, axis=0)


def tokenize(text, remove_stopwords=True):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) > 1]
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return words

def get_ngrams(tokens, n):
    if len(tokens) < n: return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def cosine_sim(v1, v2):
    if v1 is None or v2 is None: return 0.0
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def synonym_overlap(ngA, ngB):
    if not ngA or not ngB: return 0.0
    matches = 0
    for wb in ngB:
        sid_b = WORD_TO_SYN_ID.get(wb)
        for wa in ngA:
            if wa == wb or (sid_b is not None and WORD_TO_SYN_ID.get(wa) == sid_b):
                matches += 1; break
    return matches / len(ngB)

def jaccard(ngA, ngB):
    sA, sB = set(ngA), set(ngB)
    inter = len(sA & sB)
    uni = len(sA | sB)
    return inter / uni if uni > 0 else 0.0

def hybrid_sim(ngA, ngB, model, alpha=0.50, beta=0.35, gamma=0.15):
    vA = model.ngram_vector(ngA)
    vB = model.ngram_vector(ngB)
    sem = cosine_sim(vA, vB)
    syn = synonym_overlap(ngA, ngB)
    lex = jaccard(ngA, ngB)
    return alpha * sem + beta * syn + gamma * lex

def tfidf_weight(ngram, freq_map, total):
    count = freq_map.get(ngram, 0)
    if count == 0 or total == 0: return 1.5
    return round(1.0 + (1.0 - min((count/total)*5, 1.0)), 3)

def positional_weight(pos, total, n):
    if total <= 1: return n
    rel = pos / (total - 1)
    importance = 1.0 + 0.5 * (1.0 - abs(rel - 0.5) * 2)
    return max(1, round(importance * n))

def knapsack(items, capacity):
    n = len(items)
    if n == 0 or capacity <= 0: return 0.0, []
    dp = [[0.0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        val, wt, _ = items[i-1]
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]
            if wt <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-wt] + val)
    selected, w = [], capacity
    for i in range(n, 0, -1):
        val, wt, _ = items[i-1]
        if abs(dp[i][w] - dp[i-1][w]) > 1e-9:
            selected.append(i-1); w -= wt
    selected.reverse()
    return dp[n][capacity], selected


def analyze_documents(docA: str, docB: str):
    """Main analysis function — returns full result dict"""
    tokA_clean = tokenize(docA, True)
    tokB_clean = tokenize(docB, True)
    tokA_raw   = tokenize(docA, False)
    tokB_raw   = tokenize(docB, False)

    if len(tokA_clean) < 2 or len(tokB_clean) < 2:
        return {"error": "Documents too short for analysis."}

    # Train Word2Vec
    sentences = [tokA_raw, tokB_raw, tokA_clean, tokB_clean]
    for tok in [tokA_raw, tokB_raw]:
        for i in range(0, len(tok)-2, 2):
            sentences.append(tok[i:i+5])

    model = Word2Vec(dim=50, window=3, epochs=300, lr=0.03)
    model.train(sentences)

    thresholds = {1: 0.40, 2: 0.38, 3: 0.35}
    scores = []
    all_matches = []

    for n in [1, 2, 3]:
        ngramsA = get_ngrams(tokA_clean, n)
        ngramsB = get_ngrams(tokB_clean, n)
        if not ngramsA or not ngramsB:
            scores.append(0.0)
            continue

        freqA = Counter(ngramsA)
        items = []
        for pos, ngB in enumerate(ngramsB):
            best_sim, best_ngA = 0.0, None
            for ngA in ngramsA:
                sim = hybrid_sim(ngA, ngB, model)
                if sim > best_sim:
                    best_sim, best_ngA = sim, ngA
            if best_sim >= thresholds[n] and best_ngA is not None:
                boost  = tfidf_weight(best_ngA, freqA, len(ngramsA))
                weight = positional_weight(pos, len(ngramsB), n)
                items.append((best_sim * boost, weight, {
                    "ngB": " ".join(ngB),
                    "ngA": " ".join(best_ngA),
                    "sim": round(best_sim * 100, 1),
                    "rare": boost > 1.3,
                    "n": n
                }))

        capacity = sum(w for _, w, _ in items)
        max_val, sel_idx = knapsack(items, capacity)
        score = min((max_val / len(ngramsB)) * 100.0, 100.0)
        scores.append(score)

        for idx in sel_idx:
            all_matches.append(items[idx][2])

    weights = [0.25, 0.45, 0.30]
    final = min(sum(s*w for s,w in zip(scores, weights)), 100.0)

    # Find highlighted words (words in docB that matched docA)
    matched_words_A = set()
    matched_words_B = set()
    for m in all_matches:
        for w in m["ngA"].split(): matched_words_A.add(w)
        for w in m["ngB"].split(): matched_words_B.add(w)

    def highlight_text(text, matched_words):
        words = text.split()
        result = []
        for word in words:
            clean = re.sub(r"[^a-z]", "", word.lower())
            if clean in matched_words:
                result.append(f'<mark>{word}</mark>')
            else:
                result.append(word)
        return " ".join(result)

    return {
        "score": round(final, 1),
        "verdict": verdict(final),
        "verdict_level": verdict_level(final),
        "scores_by_ngram": {
            "unigram": round(scores[0], 1),
            "bigram":  round(scores[1], 1),
            "trigram": round(scores[2], 1),
        },
        "matches": sorted(all_matches, key=lambda x: x["sim"], reverse=True)[:20],
        "total_matches": len(all_matches),
        "vocab_size": len(model.vocab),
        "highlighted_source": highlight_text(docA, matched_words_A),
        "highlighted_suspect": highlight_text(docB, matched_words_B),
    }

def verdict(score):
    if score >= 70: return "HIGH PLAGIARISM"
    if score >= 40: return "MODERATE PLAGIARISM"
    if score >= 15: return "LOW SIMILARITY"
    return "LIKELY ORIGINAL"

def verdict_level(score):
    if score >= 70: return "high"
    if score >= 40: return "moderate"
    if score >= 15: return "low"
    return "original"
