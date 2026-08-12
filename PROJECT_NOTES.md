# pdf-qa — full project flow

## What it does

Ask a question about a PDF, get an answer grounded in that PDF's content.
The model can't see your documents, so you find the relevant parts
yourself and paste them into the prompt. That's RAG.

## The pipeline

    python qa.py yourfile.pdf
              |
              v
    main()
      |
      |-- extract_text(filename)
      |     PdfReader opens the PDF, loop over pages, glue text together
      |     returns: one long string
      |
      |-- chunk_text(text, size=1500)
      |     slice the string into 1500-char pieces
      |     returns: a LIST of strings — 21 chunks for my test PDF
      |
      |-- for each chunk: embed(chunk)
      |     each chunk -> 3072 numbers
      |     collected into: vectors[]
      |     vectors[0] pairs with chunks[0], by position
      |
      |-- retrieve(question, chunks, vectors, top_n)
      |     embed the question too -> q_vector
      |     for every chunk vector: similarity(q_vector, vectors[i])
      |     store (score, i) tuples, sort descending, take top n
      |     returns: the TEXT of the best-matching chunks
      |
      |-- answer(question, retrieved)
      |     join those chunks into one context block
      |     one API call: instruction + context + question
      |
      v
    print(the answer)

## The shape

    one PDF -> one string -> many chunks -> many vectors
                                                  |
                                    question vector compared to all
                                                  |
                                            top 3-5 chunks
                                                  |
                                             one answer

Two phases: INDEXING (everything up to vectors) happens once per document.
QUERYING (retrieve + answer) happens per question.
My version redoes both every run — nothing is persisted.

## Questions I had, and the answers

**Why embeddings at all — couldn't I just paste the whole PDF in?**
Yes, for short documents. It breaks on long ones (context limits, and
quality degrades in the middle), it re-sends the whole document for every
question, and it burns rate limits. Embeddings let you send only the
relevant 3 chunks.

**What IS an embedding?** Text converted into a vector of numbers that
represents its MEANING, positioned so similar meanings sit numerically
close. Not just "text as numbers" — the meaning part is what matters.

**What does one number mean?** Nothing. Like a coordinate — latitude
alone locates nothing. Unlike lat/long, the axes here have no assigned
meaning at all; the model learned them and nobody knows what dimension 47
represents. Meaning lives in the whole vector as a position.

**Why do chunks and vectors pair by position?** Because the loop appends
in order. Retrieval finds an INDEX, not text — I discover vector 7 scores
highest, then use index 7 to get chunks[7], the actual words. The index is
the bridge. Fragile: reorder one list and it breaks silently.

**Why is the question embedded too?** You can't compare text to numbers.
Both sides have to be vectors before you can measure distance.

**How does it know which chunk is relevant?** It doesn't, in advance.
It compares the question vector against ALL of them and discovers the
answer by ranking. That's searching, not looking up.

**Why cosine similarity?** dot(v1,v2) / (norm(v1) \* norm(v2)). The dot
product grows with vector length, so dividing by both norms strips
magnitude out and leaves pure direction. Range -1 to 1.

**Why is 87% (my paper's accuracy figure) hard to find?** Embeddings rank
by TOPIC, not by whether a chunk answers the question. The section heading
"6. EVALUATION AND RESULTS" outranked the chunk containing the number.

## What I measured

| Document type                  | Top 3 scores          | Spread |
| ------------------------------ | --------------------- | ------ |
| Layout-heavy (tables, columns) | 0.614 / 0.586 / 0.582 | 0.03   |
| Prose-heavy (research paper)   | 0.674 / 0.617 / 0.611 | 0.06   |

Flat spread = retrieval found nothing and is picking near-randomly.
Scores cluster narrowly either way, so judge by RANK, never by an
absolute threshold like "accept above 0.8".

Answer chunk ranked 4th at 0.610, below an unrelated chunk at 0.611.
top_n=3 missed it; top_n=5 found it.

## Known limitations

- Only works well on prose. Layout-heavy PDFs extract as fragments —
  that's pypdf losing reading order, upstream of anything I built.
- Never says "not found" — always returns top n. The prompt instruction
  is the only defence against answering from irrelevant context.
- Chunks cut mid-sentence; a heading and its content landed in different
  chunks. Overlap would fix this.
- Vectors recomputed every run; nothing persisted.

## Bugs I hit

- `scores.append(score, i)` — append takes ONE argument; needs double
  brackets to make it one tuple
- `q_vector = question` — passed a string into similarity(). numpy
  `dtype('<U32')` error always means text where numbers were expected
- Commenting out a function definition breaks the calls to it
