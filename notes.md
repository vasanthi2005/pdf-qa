**Embedding** -Converting text into vector of numbers that they capture meaning,such texts with similar meaning produce numerically closer vectors.

> Individual numbers mean nothing — like a coordinate, a single value doesn't locate anything on its own. Unlike latitude and longitude, though, the axes here have no assigned meaning; the model learned them from data and nobody knows what any single dimension represents. Meaning lives in the whole vector as a position, not in any one number.

**Chunks and vectors pair by position** — the loop appends in order, so
`chunks[0]` produces `vectors[0]`, and so on. Retrieval finds an index,
not text: you discover vector 7 scores highest, then use index 7 to get
`chunks[7]`, which is the actual text you send to the model. The index
is the bridge between the numbers you search and the words you use.

**Retrieval (pdf-qa)**: small chunks, 1500 chars. A large chunk covers many
topics, so its embedding is a blurry average of all of them and doesn't
match any specific question well. Small chunks are usually about one
thing, so their vectors are precise. You also send less irrelevant text
to the model.

Too small is bad too — a 100-char chunk loses the surrounding context
needed to make sense of it. Typical range is 500-2000.

**Embedding models are not chat models.** `gemini-embedding-001` only produces vectors — it can't answer questions. Different endpoint too:
`embed_content()` rather than `generate_content()`.

**Vector dimension is fixed by the model, not the input.** Every chunk produces exactly 3072 numbers whether it's one word or 1500 characters.
That's what makes them comparable.

**Batching** — `embed_content` accepts a list, so several texts can go
in one call and come back in the same order. Fewer round trips, and it
counts as one request against the rate limit. Currently looping one at
a time; worth switching if rate limits bite.

## Cosine similarity

Measures how similar two vectors are by the angle between them,
ignoring their length.

    similarity = dot(v1, v2) / (norm(v1) * norm(v2))

**Dot product** — multiply the two vectors element by element, sum the
results. One number out. Large when they point the same way.

**Norm (L2/Euclidean)** — the length of one vector: square every
number, sum, square root. Pythagoras extended to n dimensions.
Computed separately for each vector; the denominator is just those two
numbers multiplied.

**Why divide** — the dot product grows with vector length, so a longer
chunk could score higher just for being long. Dividing by both norms
strips magnitude out, leaving pure direction.

**Range** -1 to 1, since that's the range of cosine. 1 = same
direction, 0 = unrelated, -1 = opposite.

**In practice, though** — my test gave 0.76 for related sentences and
0.54 for unrelated ones. Real scores cluster in a narrow band, not
across the full range. So compare relatively (which chunk scores
highest) rather than against a fixed threshold.

**Why this matters** — "the cat sat on the mat" and "a feline rested
on the rug" share no words, but scored 0.76. Keyword search finds
nothing; embeddings find the meaning. That's the whole reason for the
vector machinery.

**NumPy** — `np.dot()` and `np.linalg.norm()` do these operations
across all 3072 numbers in one call. `linalg` = linear algebra. Doing it with Python loops would be much slower.

## Retrieval (RAG)

**The full chain** — chunk the document, embed each chunk, embed the
question, score the question against every chunk vector, take the top
n, send those chunks plus the question to the model.

**retrieve()**

    def retrieve(question, chunks, vectors, top_n=3):
        q_vector = embed(question)          # question must be embedded first
        scores = []
        for i in range(len(vectors)):
            scores.append((similarity(q_vector, vectors[i]), i))
        scores.sort(reverse=True)
        return [(chunks[i], score) for score, i in scores[:top_n]]

- `scores` is a list of TUPLES: (score, index). Pairing them means the
  index travels with its score through the sort — otherwise sorting
  destroys the link between a score and the chunk that produced it
- score goes FIRST in the tuple because sort compares element 0
- `scores[:top_n]` = slicing, same as text[:500]
- `for score, i in ...` = tuple unpacking, splits each pair into two names
- `top_n=3` is a default parameter — configurable without editing the code

**answer()** — joins the retrieved chunks into one context block, then
one API call. Prompt must instruct "say so if the answer isn't in the
context", because RAG ALWAYS returns something — there is no "not found".
Without that instruction the model answers confidently from irrelevant text.

## Diagnostics — what I actually learned

**Score spread is the signal, not the absolute value.**

- Layout-heavy PDF (tables, columns): 0.614 / 0.586 / 0.582 — a 0.03
  spread, essentially random. Retrieval found nothing.
- Prose-heavy PDF: 0.674 / 0.617 / 0.611 — 0.06 spread, correct section
  ranked first.
  Scores cluster in a narrow band either way, so never set a fixed
  threshold like "accept above 0.8". Compare relatively.

**Retrieval quality depends on extraction quality.** Same code, two
PDFs, completely different results. Broken text -> broken embeddings.
Tables and multi-column layouts extract as fragments with no sentence
structure, so their vectors don't resemble a natural-language question.

**Embeddings rank by topic, not by answerhood.** The chunk containing
the answer (87) ranked 4th at 0.610, while an unrelated chunk about
intrusion detection scored 0.611. A section HEADING outranked the
section's actual content, because the heading reads as topically
relevant. Re-ranking is the proper fix; raising top_n to 5 worked here.

**top_n is a quality lever, not just a setting.** 3 missed the answer,
5 found it. But more chunks = more irrelevant text in the prompt, so
it's a tradeoff, not a free win.

**Chunk boundaries cost answers.** The 1500-char cut split section 6's
heading from its content. Overlap (start = start + size - 200) makes
chunks share text with their neighbours so boundary information appears
in both. Not implemented.

## Gotchas hit

- `scores.append(score, i)` fails — append takes ONE argument. Wrap in
  an extra pair of brackets to make it one tuple: `append((score, i))`
- `q_vector = question` passes a STRING into similarity(). numpy error
  `dtype('<U32')` means text where numbers were expected — always a
  sign something wasn't embedded
