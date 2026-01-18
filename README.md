# 我的第一个 NLP 项目

这是一个文本分类 baseline

Results

We evaluate the TF-IDF + Logistic Regression baseline on the IMDB test set (25,000 samples).

Test Accuracy

Accuracy: 0.8768

Confusion Matrix (Test)

	Predicted Negative	Predicted Positive
True Negative	10936	1564
True Positive	1517	10983

The confusion matrix shows that the model performs similarly on both classes, with comparable numbers of false positives and false negatives. This indicates no strong class bias.

Classification Report (Test)

Negative class (0): precision 0.878, recall 0.875, F1-score 0.877

Positive class (1): precision 0.875, recall 0.879, F1-score 0.877

Overall, this baseline achieves strong and balanced performance for a linear model using bag-of-words style features.

Interpretation

This baseline demonstrates that TF-IDF representations combined with Logistic Regression can already capture a significant amount of sentiment information from text. Most classification errors occur on reviews with ambiguous or mixed sentiment, which is expected for a linear model without contextual understanding. This result provides a solid and reproducible baseline for future improvements such as hyperparameter tuning or replacing the model with neural approaches (e.g., BERT).