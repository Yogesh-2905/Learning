# Medical Record RAG System

A Retrieval-Augmented Generation (RAG) application that answers
questions about a medical-record PDF by retrieving relevant passages and
generating answers grounded in those passages.

The project demonstrates a complete document-to-answer workflow: PDF
extraction, text cleaning, chunking, embedding generation, vector
storage, semantic retrieval, query rewriting, context construction,
grounded answer generation, citations, and evaluation.

> **Important:** This project uses a synthetic 22-page medical-record
> PDF for learning and demonstration. It is not a clinical
> decision-support tool and must not be used to make real patient-care
> decisions.

------------------------------------------------------------------------

## 1. Project at a glance

  ---------------------------------------------------------------------
  Item                               Implementation
  ---------------------------------- ----------------------------------
  Input                              One synthetic 22-page
                                     medical-record PDF

  PDF extraction                     PyMuPDF

  Text processing                    Python

  Chunking strategies tested         Fixed, sentence, recursive

  Chunk sizes tested                 300, 600, 900 tokens

  Chunk overlap                      50 tokens

  Embedding model                    `BAAI/bge-small-en-v1.5`

  Embedding dimension                384

  Embedding generation               FastEmbed

  Vector database                    ChromaDB

  Similarity metric                  Cosine similarity

  Selected chunking configuration    `recursive_300`

  Selected retrieval Top-K           8

  Reranking                          Disabled after evaluation

  Query rewriting                    LLM-based

  Answer generation                  OpenAI Responses API, model
                                     configured through environment
                                     variables

  Evaluation questions               28 (24 answerable and 4
                                     unanswerable)
  ---------------------------------------------------------------------

## 2. What problem does this project solve?

A medical-record PDF can contain many pages of clinical notes, imaging
reports, prescriptions, follow-up records, and other information.
Finding a specific fact manually can take time.

This system lets a user ask a natural-language question, for example:

> What abnormality was found at L4-L5 on the March 6 lumbar MRI?

Instead of sending the entire PDF directly to the language model, the
system:

1.  Rewrites the question into a search-friendly form.
2.  Converts the rewritten question into an embedding vector.
3.  searches the vector database for semantically similar chunks.
4.  Supplies the retrieved passages and their source metadata to the
    LLM.
5.  Generates an answer using the retrieved context.
6.  Includes a document/page citation, or returns `I don't know` when
    the context is insufficient.

This is the central idea of **Retrieval-Augmented Generation (RAG)**:
retrieve relevant information first, then generate an answer grounded in
that information.

------------------------------------------------------------------------

## 3. End-to-end architecture

The following diagram shows the complete path from the source PDF to the
final answer.

``` mermaid
flowchart TD
    A([Start: Medical-record PDF]) --> B[Extract text from 22 PDF pages<br/>PyMuPDF]
    B --> C[Clean extracted text<br/>Normalize whitespace and line endings]
    C --> D[Split text into chunks<br/>Compare 9 chunking configurations]
    D --> E[Generate document embeddings<br/>FastEmbed + BGE-small-en-v1.5]
    E --> F[Store vectors, chunk text and metadata<br/>ChromaDB]

    G([User asks a question]) --> H[Rewrite the question<br/>LLM query rewriting]
    H --> I[Add BGE query instruction prefix]
    I --> J[Generate and normalize query embedding]
    J --> K[Cosine similarity search in ChromaDB]
    F --> K

    K --> L[Retrieve Top-8 chunks]
    L --> M[Keep original similarity ranking<br/>Reranking OFF]
    M --> N[Build context with source metadata<br/>Document ID, page and section]
    N --> O[Send original question + context + instructions<br/>to the LLM]
    O --> P{Does the context contain<br/>enough evidence?}
    P -- Yes --> Q[Generate grounded answer]
    Q --> R[Include citation<br/>[doc_id, page]]
    P -- No --> S[Return: I don't know.]
    R --> T([Final response])
    S --> T
```

### How to explain this workflow in a presentation

You can explain the diagram in two parts:

**Part A --- Prepare the knowledge base (offline process)**

-   Extract text from the PDF.
-   Clean the text while preserving clinical details.
-   Divide the text into smaller chunks.
-   Convert each chunk into a 384-dimensional embedding.
-   Store the embeddings, original chunk text, and metadata in ChromaDB.

**Part B --- Answer a user's question (query-time process)**

-   Receive and rewrite the user's question.
-   Convert the rewritten question into an embedding.
-   Search ChromaDB using cosine similarity.
-   Retrieve the eight most similar chunks.
-   Prepare context with document and page information.
-   Ask the LLM to answer only from that context.
-   Return the answer with citations, or the fallback response if
    evidence is insufficient.

------------------------------------------------------------------------

## 3A. Project decision-tree workflow

This is the decision-oriented view of the system. It shows the questions
the pipeline checks, the route taken for each outcome, and what happens
next. It is intended as a presentation aid alongside the end-to-end
architecture diagram above.

``` mermaid
flowchart TD
    START([Start]) --> PDF[Load synthetic medical-record PDF]
    PDF --> PDFQ{Can the PDF be read?}
    PDFQ -- No --> PDFERR[Stop and report PDF extraction error]
    PDFQ -- Yes --> EXTRACT[Extract text from all 22 pages]
    EXTRACT --> TEXTQ{Are pages empty or text unusable?}
    TEXTQ -- Yes --> CLEANERR[Inspect extraction and correct source or extraction]
    TEXTQ -- No --> CLEAN[Clean text while preserving clinical meaning]
    CLEAN --> CHUNK[Create chunks using tested strategy and size]
    CHUNK --> CHUNKQ{Which chunk configuration is being evaluated?}
    CHUNKQ --> EXP[Generate embeddings for each of 9 configurations]
    EXP --> EMBQ{Is the chunk within the embedding model input limit?}
    EMBQ -- No --> WINDOW[Split into overlapping token windows]
    WINDOW --> VEC[Embed windows, combine to one chunk vector, normalize]
    EMBQ -- Yes --> VEC
    VEC --> STORE[Store vectors, chunk text and metadata in ChromaDB]
    STORE --> READY([Knowledge base ready])

    READY --> USERQ([User submits a question])
    USERQ --> REWRITE[LLM rewrites question for search]
    REWRITE --> QEMB[Add BGE query prefix and create normalized query vector]
    QEMB --> SEARCH[Cosine similarity search in ChromaDB]
    SEARCH --> TOPK[Retrieve Top-8 chunks in similarity order]
    TOPK --> RERANKQ{Is reranking selected?}
    RERANKQ -- Yes --> RERANK[Apply reranker]
    RERANKQ -- No --> KEEP[Keep original similarity ranking]
    RERANK --> CONTEXT[Build context with doc ID, page and section]
    KEEP --> CONTEXT
    CONTEXT --> EVIDENCEQ{Does retrieved context contain enough evidence?}
    EVIDENCEQ -- No --> FALLBACK[Return exactly: I don't know.]
    EVIDENCEQ -- Yes --> LLM[Send original question and context to LLM]
    LLM --> ANSWER[Generate answer using context only]
    ANSWER --> CITATION[Format source citation as doc ID and page]
    FALLBACK --> END([Return response to user])
    CITATION --> END

    style PDFERR fill:#FCE8E6,stroke:#C5221F,color:#202124
    style CLEANERR fill:#FCE8E6,stroke:#C5221F,color:#202124
    style FALLBACK fill:#FFF4CE,stroke:#B7791F,color:#202124
    style END fill:#E6F4EA,stroke:#188038,color:#202124
    style READY fill:#E6F4EA,stroke:#188038,color:#202124
```

### Decisions made in this project

  -----------------------------------------------------------------------
  Decision point          Project decision        Why
  ----------------------- ----------------------- -----------------------
  Chunking strategy and   `recursive_300`         Highest nDCG (0.827) in
  size                                            the chunking
                                                  comparison, with 1.000
                                                  Hit Rate and 0.965
                                                  Recall

  Number of retrieved     Top-K = 8               Reached 1.000 Recall
  results                                         and 0.834 nDCG; larger
                                                  K did not improve nDCG

  Reranking               OFF                     Reranking reduced MRR
                                                  from 0.810 to 0.743 and
                                                  nDCG from 0.834 to
                                                  0.783

  Evidence sufficient for If no, return           Avoids generating an
  answer?                 `I don't know.`         answer when the
                                                  retrieved context does
                                                  not support it

  Evidence sufficient for If yes, generate with   Keeps the response
  answer?                 citations               grounded in retrieved
                                                  medical-record passages
  -----------------------------------------------------------------------

**How to present this decision tree:** Start at the top and follow the
arrows. Explain that the upper branch prepares the searchable
medical-record knowledge base once. The lower branch runs for each user
question. At the final evidence decision, the system either returns the
safe fallback or generates an answer with a source citation.

## 4. Data preparation

### 4.1 PDF extraction

-   Extracted text from all 22 pages using PyMuPDF.
-   Saved the extracted page content to
    `data/processed/extracted_pages.json`.
-   Checked that no pages were empty.

### 4.2 Text cleaning

The preprocessing step normalized line endings, non-breaking spaces, and
repeated whitespace.

Clinical meaning was deliberately preserved. Important details such as
dates, laterality (left/right), medication details, punctuation,
negation, and `[illegible]` markers were not intentionally removed.

Output: `data/processed/cleaned_pages.json`.

### 4.3 Chunking experiments

Three chunking strategies were compared:

  ---------------------------------------------------------------------
  Strategy                           Description
  ---------------------------------- ----------------------------------
  Fixed                              Splits text into token-sized
                                     windows

  Sentence                           Uses sentence boundaries when
                                     creating chunks

  Recursive                          Splits text progressively while
                                     trying to preserve meaningful text
                                     boundaries
  ---------------------------------------------------------------------

Each strategy was tested with three chunk sizes and a 50-token overlap.

  Configuration     Number of chunks
  --------------- ------------------
  fixed_300                       56
  fixed_600                       32
  fixed_900                       22
  sentence_300                    59
  sentence_600                    32
  sentence_900                    22
  recursive_300                   59
  recursive_600                   32
  recursive_900                   22

**Selected configuration: `recursive_300`.**

It achieved the strongest overall balance in the measured retrieval
metrics, including the highest nDCG among the nine configurations.

------------------------------------------------------------------------

## 5. Embeddings and vector database

### 5.1 Embedding model

The project uses:

`BAAI/bge-small-en-v1.5`

-   Embedding dimension: 384
-   Embedding generation library: FastEmbed
-   Document vectors and query vectors are L2-normalized.
-   The BGE search instruction prefix used for queries is:

``` text
Represent this sentence for searching relevant passages:
```

The embedding model converts text into a numerical vector. Texts with
related meanings can have vectors that are close to each other in the
embedding space, which makes semantic retrieval possible even when the
question and source passage do not use exactly the same words.

### 5.2 Long-chunk handling

The embedding implementation respects the model's token input limit.
Long chunks are divided into smaller overlapping windows, embedded, and
combined into one vector for the original chunk.

### 5.3 ChromaDB

ChromaDB stores:

-   Chunk embeddings
-   Original chunk text
-   Metadata such as `doc_id`, `page`, and `section`

Cosine distance is used for vector search. The retrieval code converts
the returned distance to a similarity value using:

``` python
cosine_similarity = 1 - cosine_distance
```

A higher similarity value indicates a closer vector match.

------------------------------------------------------------------------

## 6. Retrieval experiments and decisions

The project compared chunking strategies, Top-K values, and reranking.

### 6.1 Chunking comparison

The evaluation used 24 answerable questions and measured page-level
retrieval. Retrieved chunks were deduplicated by page before evaluating
the top five pages.

  ---------------------------------------------------------------------------
  Configuration      Hit Rate   Precision      Recall         MRR        nDCG
  --------------- ----------- ----------- ----------- ----------- -----------
  fixed_300             1.000       0.242       0.951       0.774       0.815

  fixed_600             0.958       0.250       0.938       0.800       0.793

  fixed_900             0.917       0.242       0.903       0.819       0.798

  sentence_300          1.000       0.250       0.965       0.772       0.815

  sentence_600          0.917       0.242       0.896       0.833       0.806

  sentence_900          0.917       0.242       0.903       0.819       0.798

  recursive_300     **1.000**   **0.250**   **0.965**       0.810   **0.827**

  recursive_600         0.917       0.242       0.896   **0.875**       0.822

  recursive_900         0.917       0.242       0.903       0.819       0.798
  ---------------------------------------------------------------------------

**Why choose `recursive_300`?**

It achieved the highest nDCG (0.827), while also achieving a 1.000 Hit
Rate and 0.965 Recall. Although `recursive_600` had a higher MRR,
`recursive_300` provided the strongest overall balance across the
selected retrieval measures.

### 6.2 Top-K comparison

Using `recursive_300`, the experiment compared different values of K.
This evaluation ranked unique pages after deduplicating retrieved
chunks.

     K    Hit Rate   Precision      Recall     MRR        nDCG
  ---- ----------- ----------- ----------- ------- -----------
     1       0.708       0.708       0.611   0.708       0.673
     3       0.917       0.333       0.826   0.792       0.764
     5       1.000       0.250       0.965   0.810       0.827
     8   **1.000**       0.167   **1.000**   0.810   **0.834**
    10       1.000       0.133       1.000   0.810       0.834
    15       1.000       0.089       1.000   0.810       0.834

**Why choose Top-K = 8?**

-   It reached 1.000 Hit Rate and 1.000 Recall in this evaluation.
-   It achieved the highest nDCG (tied with K=10 and K=15).
-   Increasing K beyond 8 did not improve Recall or nDCG.
-   Therefore, K=8 was selected instead of retrieving more passages
    unnecessarily.

**Important implementation distinction:** The experiment above
deduplicated pages for page-level evaluation. The current live
`retrieve_chunks()` function returns the top eight chunks, not
necessarily eight unique pages. Multiple chunks can therefore come from
the same page.

### 6.3 Reranking comparison

The reranker `jinaai/jina-reranker-v1-tiny-en` was tested.

  Configuration     Hit Rate   Precision   Recall         MRR        nDCG
  --------------- ---------- ----------- -------- ----------- -----------
  Reranker OFF         1.000       0.167    1.000   **0.810**   **0.834**
  Reranker ON          1.000       0.167    1.000       0.743       0.783

**Decision: Reranking OFF.**

Reranking did not improve Hit Rate, Precision, or Recall, and it reduced
MRR and nDCG. The final pipeline therefore keeps the original
cosine-similarity ranking. This also avoids the additional reranking
computation.

------------------------------------------------------------------------

## 7. Final RAG pipeline

The selected query-time pipeline is:

1.  **Query rewriting:** An LLM rewrites the user's question into a
    search-friendly query while preserving its meaning and important
    medical details.
2.  **Query embedding:** FastEmbed generates a normalized
    384-dimensional BGE query vector using the BGE query prefix.
3.  **Semantic retrieval:** ChromaDB searches the stored document
    vectors using cosine similarity.
4.  **Top-K retrieval:** The system retrieves eight chunks.
5.  **Reranking:** Disabled based on the experiment results.
6.  **Context preparation:** Retrieved chunk text is combined with its
    document ID, page, and section.
7.  **Grounded generation:** The LLM receives the original question,
    retrieved context, and instructions to use only that context.
8.  **Citations:** The answer cites supporting sources in the format
    `[doc_id, page]`.
9.  **Fallback:** If the supplied context is insufficient, the LLM is
    instructed to respond exactly `I don't know.`

### Why keep the original question for generation?

The rewritten query is used to improve retrieval. The original question
is still sent to the LLM so that the generated answer addresses what the
user actually asked, rather than answering only the shortened search
query.

------------------------------------------------------------------------

## 8. Evaluation results

### 8.1 Retrieval metrics

The selected retrieval configuration (`recursive_300`, page-level
evaluation at K=8) achieved:

  ---------------------------------------------------------------------
  Metric                                    Result Interpretation
  -------------------- --------------------------- --------------------
  Hit Rate@8                                 1.000 At least one
                                                   relevant page was
                                                   retrieved for every
                                                   evaluated answerable
                                                   question

  Precision@8                                0.167 On average, 16.7% of
                                                   the eight retrieved
                                                   unique-page
                                                   positions were
                                                   relevant under the
                                                   evaluation labels

  Recall@8                                   1.000 All labelled
                                                   relevant pages were
                                                   retrieved across the
                                                   evaluated answerable
                                                   questions

  MRR                                        0.810 Relevant evidence
                                                   generally appeared
                                                   near the top of the
                                                   ranked results

  nDCG@8                                     0.834 The ranking placed
                                                   relevant pages
                                                   relatively high,
                                                   accounting for
                                                   relevance grades
  ---------------------------------------------------------------------

**Why are these results useful?**

-   A Hit Rate of 1.000 means the system found at least one relevant
    page for every answerable test question.
-   Recall of 1.000 means it retrieved all labelled relevant pages in
    this evaluation.
-   MRR of 0.810 indicates that the first relevant result was usually
    ranked near the top.
-   nDCG of 0.834 indicates good ranking quality while considering
    graded relevance.
-   Precision is lower because the system retrieves a larger set of
    pages to achieve high recall. This is a trade-off: retrieving more
    evidence can improve coverage but also includes irrelevant results.

These values describe the project's evaluation dataset. They do not
guarantee the same performance on other medical records or questions.

### 8.2 Answer-generation and latency evaluation

The answer evaluation dataset contains 28 questions:

-   24 answerable questions
-   4 unanswerable questions

All 28 executions completed without an exception.

  Metric                                 Result
  ---------------------------- ----------------
  Total questions                            28
  Successful executions                      28
  Average end-to-end latency       4.30 seconds
  Minimum latency                  2.93 seconds
  Maximum latency                  9.97 seconds
  Total processing time          120.47 seconds

All four unanswerable test questions returned the expected
`I don't know` fallback during the answer-quality test.

**Important:** Successful execution means the code completed without an
exception. It does not mean every generated answer was factually
perfect. Answer correctness was also manually reviewed, and the project
should report answer quality as a qualitative/manual assessment rather
than treating execution success as an accuracy score.

The measured latency is end-to-end for the RAG call, including query
rewriting, embedding, retrieval, context construction, and LLM
generation. It can vary with network conditions, model response time,
and whether models need to be loaded.

### 8.3 Storage evaluation

  Component                            Size
  ----------------------------- -----------
  ChromaDB persistent storage      14.05 MB
  Nine NumPy embedding files      505.12 KB
  Nine metadata JSON files        611.57 KB
  Total embedding artifacts         1.09 MB
  Total `data/` directory          15.98 MB

The nine embedding configurations were retained so that the chunking
experiments could be compared and reproduced.

------------------------------------------------------------------------

## 9. Why the final design is reasonable

The final configuration was selected based on measured experiments, not
only intuition.

  ---------------------------------------------------------------------
  Decision                           Reason
  ---------------------------------- ----------------------------------
  BGE-small-en-v1.5                  Provides compact 384-dimensional
                                     embeddings for semantic retrieval

  `recursive_300`                    Best overall balance and highest
                                     nDCG in the chunking comparison

  Top-K = 8                          Achieved full recall in the
                                     evaluation; larger K did not
                                     improve nDCG

  Reranking OFF                      Reranking reduced MRR and nDCG in
                                     the experiment

  Query rewriting ON                 Converts the user's question into
                                     a search-friendly form before
                                     embedding

  Source metadata in context         Helps the LLM cite the document
                                     and page supporting its answer

  `I don't know` fallback            Reduces unsupported answers when
                                     retrieved context is insufficient
  ---------------------------------------------------------------------

The goal was not to maximize every metric independently. It was to
choose a practical configuration that balances retrieval coverage,
ranking quality, context size, and processing cost.

------------------------------------------------------------------------

## 10. Project workflow and important files

The main project files and outputs include:

``` text
Task 3 ( RAG )/
│
├── common/
│   ├── pdf_extraction.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── data/
│   ├── raw/
│   │   └── [source PDF]
│   │
│   ├── processed/
│   │   ├── extracted_pages.json
│   │   ├── cleaned_pages.json
│   │   ├── document_chunks.json
│   │   └── embeddings/
│   │       ├── fixed_300.npy
│   │       ├── fixed_600.npy
│   │       ├── fixed_900.npy
│   │       ├── sentence_300.npy
│   │       ├── sentence_600.npy
│   │       ├── sentence_900.npy
│   │       ├── recursive_300.npy
│   │       ├── recursive_600.npy
│   │       └── recursive_900.npy
│   │
│   ├── chroma_db/
│   └── evaluation/
│       ├── evaluation_questions.csv
│       ├── relevance_label.csv
│       └── rag_answer_results.csv
│
├── generate_answer.py
├── evaluate_answers.py
├── evaluate_storage.py
├── phase6_summary.py
├── evaluate_chunking.py
├── evaluate_top_k.py
├── evaluate_reranker.py
├── rebuild_index.py
├── .env
└── .gitignore
```

This is a summary of the files used during the work so far. The final
repository structure may be updated during the remaining delivery phase.

------------------------------------------------------------------------

## 11. Limitations and future improvements

1.  **Small evaluation corpus:** The system was evaluated on one
    synthetic 22-page PDF and 28 questions. Results may differ on larger
    and more varied medical-record collections.
2.  **Answer quality:** Retrieval success does not guarantee that every
    generated answer is complete or correct. Manual review remains
    important.
3.  **Chunk-level duplicates:** The live Top-K retrieval returns chunks,
    so multiple chunks may come from the same page.
4.  **Latency variation:** End-to-end response time depends on the
    embedding model, LLM service, network, and model loading.
5.  **Metadata and document scope:** The current prototype focuses on
    one synthetic document and basic metadata fields.
6.  **Not for clinical use:** The system is an educational prototype,
    not a validated medical device or substitute for professional
    judgement.

Potential future work includes evaluating on more documents, expanding
the labelled question set, measuring latency by pipeline stage, and
testing more robust citation verification.

------------------------------------------------------------------------

## 12. How to present this project

A simple presentation explanation:

> "I built a Medical Record Retrieval-Augmented Generation system.
> First, I extracted and cleaned text from a 22-page synthetic
> medical-record PDF. Then I experimented with three chunking strategies
> and three chunk sizes. I generated 384-dimensional embeddings using
> BGE-small-en-v1.5 through FastEmbed and stored them in ChromaDB.
>
> When a user asks a question, the system rewrites it, embeds it, and
> retrieves the eight most similar chunks using cosine similarity. I
> tested reranking, but it reduced MRR and nDCG, so I disabled it. The
> retrieved passages and their source metadata are passed to the LLM,
> which generates a grounded answer with document and page citations. If
> the context is insufficient, the system returns 'I don't know'.
>
> I evaluated retrieval using Hit Rate, Precision, Recall, MRR, and
> nDCG. I also tested 28 questions and measured latency and storage. The
> selected configuration achieved 1.000 Hit Rate, 1.000 Recall, 0.810
> MRR, and 0.834 nDCG at page-level Top-8 evaluation."

------------------------------------------------------------------------

## 13. Conclusion

This project demonstrates how semantic retrieval and LLM-based
generation can be combined to build a question-answering system over
medical-record text.

The key learning outcomes are:

-   Preparing PDF data for retrieval.
-   Comparing chunking strategies and chunk sizes.
-   Generating and storing embeddings.
-   Performing semantic search with a vector database.
-   Evaluating retrieval quality using standard ranking metrics.
-   Integrating query rewriting and grounded LLM generation.
-   Providing source citations and an insufficient-evidence fallback.
-   Measuring response latency and storage requirements.

The final selected retrieval setup is `recursive_300`, BGE-small-en-v1.5
(384 dimensions), Top-K 8, and reranking disabled.

**Project status:** Core RAG pipeline and Phase 6 evaluation completed.
Final packaging, repository preparation, and demo deliverables are the
next project phase.
