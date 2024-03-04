# LLM Questionnaire

## Running
To run the [OUS Questionnaire](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5900580/) on LLMs, execute
```bash
python questionnaire.py
```

### Default Values
By default, this will run 32 prompt structures for all 9 OUS statements on 10 models.
For any specifics on used models, prompt options and further details we defer to our [paper](../report.pdf) or have a look at the code directly.

All used default values are listed at the top of [questionnaire.py](questionnaire.py).

## Adding questions
You can simply add a column of questions in the [questions.csv](questions.csv). Then add/replace the column title in the `QUESTIONNAIRES` array in [questionnaire.py](questionnaire.py).
