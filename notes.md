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
across all 3072 numbers in one call. `linalg` = linear algebra. Doing
it with Python loops would be much slower.
