# PDF QA

Ask questions about a PDF and get answers grounded in its content.
A minimal retrieval-augmented generation (RAG) pipeline built with
Python and the Google Gemini API.

**Live demo:** https://pdf-ans.streamlit.app/

_(Free hosting — the app sleeps when idle and takes ~30 seconds to wake.)_

## Requirements

1. Chunks get embedded and stored
2. A question retrieves the most relevant chunks
3. The model answers using only those chunks

## Setup

    pip install google-genai python-dotenv pypdf numpy

Create a `.env` file in the project root:

    GOOGLE_API_KEY=your-key-here

A free key is available from Google AI Studio (aistudio.google.com).

## Usage

Supply your own PDF — none is included in this repo.

    python qa.py yourfile.pdf

The question is currently set inside `main()`. The PDF must contain
real text; scanned documents would need OCR.

## How it works

1. `pypdf` extracts text from every page
2. The text is split into 1,500-character chunks
3. Each chunk is embedded into a 3,072-dimension vector
   (`gemini-embedding-001`)
4. The question is embedded the same way
5. Cosine similarity scores the question against every chunk vector
6. The highest-scoring chunks are sent to the model with the question

Chunks are 1,500 characters here rather than the 15,000 used for
summarising in the companion project. Large chunks cover many topics,
so their embeddings become a blurry average that matches no specific
question well.

## Findings

**Retrieval quality depends on extraction quality.** The same code was
tested on two documents:

| Document                            | Top 3 scores          | Spread |
| ----------------------------------- | --------------------- | ------ |
| Layout-heavy (tables, multi-column) | 0.614 / 0.586 / 0.582 | 0.03   |
| Prose-heavy (research paper)        | 0.674 / 0.617 / 0.611 | 0.06   |

The layout-heavy PDF extracted as fragmented text with collapsed
tables, producing poor embeddings. Its flat score spread means
retrieval was effectively choosing at random. The prose document
ranked the correct section first.

**Scores cluster in a narrow band.** Even a good match scored 0.674
against an unrelated 0.611. Absolute thresholds ("accept above 0.8")
don't work; only the ranking is meaningful.

**Embeddings rank by topic, not by answerhood.** Asked for a specific
accuracy figure, the top result was the section _heading_ — topically
relevant but containing no numbers. The chunk holding the actual answer
ranked 4th at 0.610, below an unrelated chunk at 0.611. Raising
`top_n` from 3 to 5 produced the correct answer.

**Chunk overlap did not help.** I hypothesised that the answer chunk ranked
poorly because a 1,500-character cut split section headings from their content,
so I added 200-character overlap between chunks and re-measured with the same
question and document.

|                  | Top 3 scores          | Answer chunk  |
| ---------------- | --------------------- | ------------- |
| No overlap       | 0.674 / 0.617 / 0.611 | rank 4, 0.610 |
| 200-char overlap | 0.657 / 0.616 / 0.612 | rank 4, 0.610 |

No improvement, and the top score fell slightly. This ruled out chunk
boundaries as the cause and confirmed the topical-ranking problem instead: the
top-scoring chunk contains the phrase "gauge the prediction accuracy of its
machine learning models" but no figure, while the chunk stating 87 never uses
the word "accuracy" near it. Re-ranking or hybrid keyword search would be the
appropriate fix, not different chunking.

## Limitations

**No "not found" signal.** Retrieval always returns its top n chunks
regardless of whether any are relevant. The prompt instructs the model
to say when the context doesn't contain the answer — without that, it
answers confidently from irrelevant text.

**Chunks cut mid-sentence.** Fixed-size splitting counts characters,
not meaning. In testing, a section heading and its content landed in
different chunks, which is why the answer chunk ranked poorly.

**Vectors are recomputed on every run.** Nothing is persisted, so the
whole document is re-embedded each time. Fine for a single question,
wasteful for repeated use.

## Possible improvements

- Persist vectors in a vector database (ChromaDB) instead of two parallel lists
- Add re-ranking — a second pass scoring chunks on whether they answer
  the question rather than merely relating to it
- Batch the embedding calls instead of one request per chunk
