# Data format

Local CSV files must contain at least two columns:

```csv
text,label
"This product is excellent",positive
"یہ فلم بہت اچھی ہے",positive
```

Recommended language values are `english`, `urdu`, and `roman_urdu`. The language value is used only for reporting and does not need to be embedded in the text.

Keep raw datasets out of Git when they are large. The training scripts download Hugging Face datasets automatically when `source: huggingface` is configured.
