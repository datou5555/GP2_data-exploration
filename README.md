# Amazon Electronics Review Rating Prediction Using NLP and Logistic Regression

## 🔍 Introduction

Amazon product reviews are a rich source of textual information and carry important signals about customer satisfaction. With millions of daily reviews across product categories, businesses can gain significant insights from modeling this data. We chose the Amazon Electronics 2024 Reviews dataset from Kaggle to build a model that predicts user star ratings based solely on review text.

This task is compelling because it combines natural language understanding with a practical business application. Accurate review prediction can improve content moderation, enhance recommendation systems, and assist customer service with early issue detection.

## 📊 Figures

- Figure 1. Distribution of Review RatingsThe class imbalance is clear: the dataset is skewed toward 5-star reviews.

- Figure 2. Correlation Between Review Length and RatingA scatter plot showing weak correlation between the number of words and the rating.

- Figure 3. Average Word Length per Rating (Boxplot)Boxplots visualizing linguistic tendencies per rating group.

(Figures to be inserted in the report and notebooks.)

## 🧪 Methods

### 🔎 Data Exploration

- Dataset: Amazon Electronics Reviews 2024 - Kaggle

- ~43 million rows total

- Dropped unused columns: asin, user_id, timestamp, title, etc.

- Observed severe class imbalance (most reviews are 5-star)

- Sampled or filtered subset for practical model training due to compute constraints

### 🧼 Preprocessing

# Convert to lowercase
clean_review_df = clean_review_df.withColumn("lower_text", lower("text"))

# Tokenize, remove stopwords, stem
@udf(returnType=ArrayType(StringType()))
def tokenize_text(text):
    # ... Uses NLTK stopwords and PorterStemmer

- Additional features computed:

- - Word count

- - Token count

- - Average word length

### 🤖 Model 1: Multinomial Logistic Regression (Baseline)

- Text vectorized using CountVectorizer + TF-IDF

- Labels indexed from original ratings

- LogisticRegression parameters:

- - maxIter=20, regParam=0.1, elasticNetParam=0.0

- Train-test split: 80/20

pipeline = Pipeline(stages=[cv, idf, indexer, lr])
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

## 📈 Results

### 📊 Model 1: Logistic Regression

- Training Accuracy: 69.19%

- Test Accuracy: 68.76%

- Minimal overfitting observed

- Output includes predicted class probabilities

### Table 1. Sample Output:

Review Text

True Rating

Prediction

Probability

...

5

5.0

[0.01, 0.02, ..., 0.87]

## 💬 Discussion

### Model 1 – Logistic Regression

We selected logistic regression as our baseline due to its simplicity, interpretability, and strong performance in text classification tasks with TF-IDF features. The model achieved a balanced performance without overfitting. However, it struggled to accurately differentiate between close sentiment ratings (e.g., 3 vs. 4 stars), a known challenge in ordinal classification of subjective reviews.

One limitation we faced was class imbalance, which may have biased the model toward predicting the majority class (5 stars). Logistic regression is also limited in modeling non-linear relationships, which likely exist in real-world review data.

## ✅ Conclusion

Our first model—a multinomial logistic regression trained on TF-IDF features—achieved ~69% accuracy and exhibited generalization with minimal overfitting. The results are credible and consistent with expectations for a baseline model. We believe performance can be improved through more expressive models, better class balancing, and feature engineering. Given time, we would also explore neural models and topic modeling techniques.

## 🤝 Collaboration

Jinxin XiaoTitle: Contributor (Data Processing, Modeling, and Write-Up)Contribution: Implemented full preprocessing and modeling pipeline in PySpark. Contributed to all write-up sections, figures, and evaluations. Met weekly and collaborated on decision-making, code sharing, and validation.

Teammate 2 NameTitle: Contributor (Exploration, Model Evaluation, and Communication)Contribution: Focused on initial data analysis and exploration. Helped in writing discussion sections and interpreting model results. Coordinated meetings and helped manage GitHub repo.

Teammate 3 NameTitle: Contributor (Testing, Cleanup, and Visualization)Contribution: Assisted in cleaning data and writing visualizations. Produced scatter plots and boxplots used in results. Reviewed and edited notebooks before submission.

If someone did not participate, write: "Did not participate in the project."

## Final Accuracy (Logistic Regression): 68.76% (Test Set)

Let me know if you'd like this exported to a README.md file or if you'd like me to help generate the visualizations or GitHub structure!


