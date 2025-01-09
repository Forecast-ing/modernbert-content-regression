---
title: ModernBERT Engagement Content Regression
app_file: app.py
pinned: true
---

# ModernBERT Engagement Content Regression
### What is this?

This is an exploration of using modernBERT for the text regression task of predicting engagement metrics for text content. In this case, we are predicting the clickthrough rate (CTR) of email text content.

We will be exploring hyperparameter tuning of modernBert; and how to use it for regression, as well as comparing the results to a benchmark model.

This type of task if difficult, we can remember the quote
> “Half my advertising is wasted; the trouble is, I don't know which half”
> -John Wanamaker

We are also excluding other relevant factors such as the time of day the email is sent, the day of the week, the recipient, etc in this experiment.

This work is indebted to the work of many community members and blog posts.
- [ModernBERT Announcement](https://huggingface.co/blog/modernbert)
- [Fine-tune classifier with ModernBERT in 2025](https://www.philschmid.de/fine-tune-modern-bert-in-2025)
- [How to set up Trainer for a regression](https://discuss.huggingface.co/t/how-to-set-up-trainer-for-a-regression/12994)


Our model - [ModernBERT-Engagement-Content-Regression](https://huggingface.co/Forecast-ing/modernBERT-content-regression)
Our training notebook - [Training Notebook](https://github.com/Forecast-ing/modernbert-content-regression/blob/main/model_training.ipynb)

### Our dataset
We will be using a dataset of 548 emails where we have the text of the email `text` and the CTR we are trying to predict `labels`.

We look forward in the improvements of ModernBERT to fine-tune models specifically for each potential users email dataset. The variability of email data, as well as the small size of the dataset pose an interesting regression challenge.

### Benchmarking
We will start by using the Catboost library as a simple benchmark for text regression. For both the benchmark and the ModernBert run, we are using 'rmse' as the metric.
We recieve the following results:
| Metric | Value            |
|--------|------------------|
| MSE    | 2.552100633998035 |
| RMSE   | 1.5975295408843102 |
| MAE    | 1.1439370629666958 |
| R²     | 0.30127932054387174 |
| SMAPE  | 37.63064694052479 |

## Fitting the Modern Bert Model

### Install dependencies and activate venv
```bash
uv sync
source .venv/bin/activate
```
the following values need to be defined in the .env file
- `HUGGINGFACE_TOKEN`

### Run notebook for model fitting

```bash
uv run --with jupyter jupyter lab
```

### ModernBert Model Performance
After running hyperparameter tuning for ModernBERT, we get the following results:

| Metric | Value            |
|--------|------------------|
| MSE    | 2.4624056816101074 |
| RMSE   | 1.5692054300218654 |
| MAE    | 1.182181715965271 |
| R²     | 0.325836181640625 |
| SMAPE  | 56.61447048187256 |

We see improvements in all metrics except for SMAPE. We believe that ModernBERT would scale even better with a larger dataset; as 500 example is very low for fine-tuning and are thus happy with the performance of this evaluation.

### Who are we?
At [Forecast.ing](https://forecast.ing) we are building a platform to help users create more enriching content by automatically researching trends and generating campaign ideas with AgenticAI. 
We generate the content, and then create fine-tuned scores of how likely we think that content will succeed.

## Conclusion
We see that ModernBERT is a powerful model for text regression. We believe that with a larger dataset, we would see even better results. We are excited to see the future of ModernBERT and how it will be used for text regression.
If interested, I can be contacted at robin@forecast.ing