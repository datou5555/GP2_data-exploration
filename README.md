# GP3_Pre-Processing
Step 4: Model Update – Logistic Regression with TF-IDF

(https://github.com/datou5555/GP2_data-exploration/blob/milestone3/milestone3.ipynb)

We have updated our baseline model by embedding the text data using TF-IDF (Term Frequency-Inverse Document Frequency). This method provides a simple yet effective numerical representation of text. For this iteration, we used:

  Features (X): Tokenized text

  Labels (y): Review ratings

  Model: Multinomial Logistic Regression

This model achieved:

  Training Accuracy: 69.19%

  Test Accuracy: 68.76%

These close scores indicate the model is well-balanced and fits in the "just right" zone on the underfitting/overfitting curve.

Step 5: Future Model – Random Forest Classifier

(https://github.com/datou5555/GP2_data-exploration/blob/milestone3/milestone3.ipynb)

To build on our baseline, we are planning to implement a Random Forest Classifier. Unlike logistic regression (a linear model), Random Forest is an ensemble learning method capable of capturing non-linear relationships and feature interactions. We believe this approach will:

  Improve predictive performance

  Reduce overfitting through ensemble averaging

  Better handle the complex structure of textual review data

Step 6: Model Evaluation and Next Steps

(https://github.com/datou5555/GP2_data-exploration/blob/milestone3/milestone3.ipynb)

Our first model provided a strong starting point, but predicting five distinct rating levels introduces noise due to subjective variations in user reviews. To enhance performance, we plan to:

  Incorporate more expressive models like Random Forest

  Add new features

  Tune hyperparameters for better accuracy




