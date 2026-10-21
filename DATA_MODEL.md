# Data model (written from the task, before looking at what the corpus holds)

The task: a person describes a paper they have in mind, in their own words, and
the system returns that paper's PMID at rank 1. MRR rewards rank 1 above all.

## Article (one row per PMID, the unit of retrieval)

| field | question it answers for the searcher |
|---|---|
| pmid | which paper (the answer itself) |
| title | what the paper says it is |
| abstract | what it did and found, in its own vocabulary |
| journal | "the Science paper on..." |
| year | "the 2012 paper..." / "early work on..." |
| authors (last names, first + last author flagged) | "Doudna and Charpentier's paper..." |
| mesh (descriptors, major-topic flag) | the controlled concept when the query uses a synonym the abstract doesn't |
| chemicals, keywords | named substances / author keywords the query may use |
| pub_types | "the trial", "the review", "the meta-analysis" |
| cited_by_n (in-corpus) | "landmark" / "seminal": the known item is usually a well-cited one |
| doi | disclosure and dedupe |

Joins: Article.references -> Article.pmid gives cited_by_n (count of corpus
articles listing this PMID in their ReferenceList). Computed from the frozen
corpus only, so it is allowed under the no-egress rule.

## DevQuery (offline, never ships in the system)

| field | question |
|---|---|
| qid | |
| query | the description a searcher would type |
| pmid | the known item |
| source | `public` (BioASQ/TREC/published examples) / `template` / `local-llm` / `claude` |
| style | title-paraphrase / description-without-title / author-year / landmark |

## Reality check (filled in after parsing)


Measured 2026-10-21 over 39,167,524 parsed records: abstract 74.1%, MeSH 82.6%,
authors 98.5%, year 100%, keywords 31.5%, reference list 30.2% (418,178,302 in-corpus
references, enough to fill cited_by_n). Every field in the model is fillable from the
frozen corpus; nothing needs an outside source.
