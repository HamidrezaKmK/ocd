"""OCD — Optimized Cash Dashboard.

A fully-local, personalized credit-card statement analyzer:

  Step 1  extract    PDFs -> data/transactions_raw.csv          (ocd.extract)
  Step 2  categorize raw  -> data/transactions_categorized.csv  (ocd.classify + ocd.review)
  Step 3  report     finalized CSV -> reports/*.md + *.html      (ocd.report)

Everything runs locally; the only model is an OpenAI-compatible LLM (Ollama by default)
used in Step 2 for categorization and optionally in Step 3 for the insight summary.
"""

__version__ = "0.1.0"
