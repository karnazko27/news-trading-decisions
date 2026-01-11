# Using News Headlines for Financial Market Trading Decisions



## Table of Contents

1. [Research and Selection of Methods](#research-and-selection-of-methods)  
   1.1 [Project Objectives and Goals](#project-objectives-and-goals)  
   1.2 [GitHub Repository Structure](#github-repository-structure)  
   1.3 [Literature Review](#literature-review)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.3.1 [Predicting Buy/Hold/Sell Signals Using News Sentiment (Goal 1)](#1-predicting-buyholdsell-signals-using-news-sentiment-goal-1)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.3.2 [Recommending Stocks Based on Investor Profiles (Goal 2)](#2-recommending-stocks-based-on-investor-profiles-goal-2)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.3.3 [Building an NLP-Based Trading Bot (Goal 3)](#3-building-an-nlp-based-trading-bot-goal-3)  
   1.4 [Benchmarking](#benchmarking)  
   1.5 [Preliminary Experiments](#preliminary-experiments)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.5.1 [Objectives](#objectives)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.5.2 [Models and Techniques Tested](#models-and-techniques-tested)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.5.3 [Preprocessing](#preprocessing)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.5.4 [Key Findings](#key-findings)  
   &nbsp;&nbsp;&nbsp;&nbsp;1.5.5 [Conclusion](#conclusion)

2. [Model Implementation](#model-implementation)  
   2.1 [Framework Selection](#framework-selection)  
   2.2 [Dataset Preparation](#dataset-preparation)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.1 [Data Loading and Integration](#1-data-loading-and-integration)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.2 [Text Normalization and Tokenization](#2-text-normalization-and-tokenization)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.3 [Sentiment Labeling](#3-sentiment-labeling)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.4 [Feature Engineering](#4-feature-engineering)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.5 [Label and Feature Encoding](#5-label-and-feature-encoding)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.2.6 [Final Dataset Structure](#6-final-dataset-structure)  
   2.3 [Model Development](#model-development)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.3.1 [Goal 1: Predict Buy / Hold / Sell (Classification)](#goal-1-predict-buy--hold--sell-classification)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.3.2 [Goal 2: Recommend Stocks Based on Investor Profile](#goal-2-recommend-stocks-based-on-investor-profile)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.3.3 [Goal 3: NLP-Based Trading Bot](#goal-3-nlp-based-trading-bot)  
   2.4 [Training and Fine-Tuning](#training-and-fine-tuning)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.4.1 [Traditional Machine Learning Models](#1-traditional-machine-learning-models)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.4.2 [Deep Learning Models](#2-deep-learning-models)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.4.3 [Recommendation Models](#3-recommendation-models)  
   2.5 [Evaluation and Metrics](#evaluation-and-metrics)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.5.1 [Classification Models](#1-classification-models)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.5.2 [Recommendation Models](#2-recommendation-models)  
   &nbsp;&nbsp;&nbsp;&nbsp;2.5.3 [Trading Bot Models](#3-trading-bot-models)

3. [References (APA Style)](#references-apa-style)






# Research and Selection of Methods
---

## Project Objectives and Goals

Our project aims to apply NLP techniques to use News Headlines, analyze their content and provide concise, actionable recommendations for financial market trading decisions. This will streamline the information-gathering process for investors, enhance decision-making efficiency.

Using a dataset enriched with sentiment scores, stock fundamentals, and classification labels, we seek to streamline information overload, support data-driven investment decisions, and enable automation in financial analysis.

To achieve this, the project is structured around three primary goals:

1. Predict Buy/Hold/Sell for each stock (classification):

We aim to classify each news headline into actionable trading signals—**Buy**, **Hold**, or **Sell**. This involves:

- **Text Classification**: Core NLP task where headlines are classified based on sentiment and contextual indicators.
- **Sentiment Analysis**: Assign sentiment scores to the headlines using rule-based (e.g., VADER) and ML-based approaches to serve as predictive features.
- **Tokenization, Lemmatization & Normalization**: Preprocessing steps to convert headlines into clean, consistent features.
- **Feature Engineering**: Utilize headline metadata (e.g., word count, day of week) alongside textual features to improve classification performance.

The classification models trained on these features will allow for real-time interpretation of market-moving news.

2. Recommend stocks based on investor profile (investment style):

To personalize recommendations, we align sentiment-informed predictions with investor characteristics such as risk appetite, sector preference, and time horizon. This involves:

- **Metadata-Aware Modeling**: Integrate stock-specific features (e.g., sector, market cap, IPO year) with sentiment scores.
- **Recommendation Logic**: Build rule-based or ML-driven strategies to suggest stocks matching investor profiles (e.g., conservative vs. aggressive strategies).

This goal enables our system to deliver investor-specific trade ideas rather than generic stock suggestions.

3. Build an NLP-based trading bot:

We will implement a prototype trading bot that automates the end-to-end process: ingesting headlines, extracting meaning, and producing trade recommendations. NLP-related tasks include:

- **Text Preprocessing Pipeline**: Automate real-time processing of news headlines (tokenization, lemmatization, normalization).
- **Sequence-to-Sequence Modeling**: If used for news summarization or abstraction, this task would help compress complex headlines into simplified trade rationale.
- **Action Mapping**: Use classification results to determine trading actions.

Together, these objectives form the foundation for a comprehensive NLP-powered financial system that bridges unstructured news data with structured investment logic, enhancing both efficiency and intelligence in trading.

We used the following datasets to conduct our analyses:

- Daily Financial News for 6000+ Stocks

  - Source: Kaggle, https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv

  - Volume: ~4m articles for 6000 stocks from 2009-2020

- NASDAQ Stocks Screener
   - Source: NASDAQ, https://www.nasdaq.com/market-activity/stocks/screener?page=1&rows_per_page=25

## Github Repository Structure

To maintain a clean, modular, and scalable codebase, our GitHub repository is organized into the following directories:

### `notebooks/`
Contains exploratory and goal-specific Jupyter notebooks for all stages of the project. Each notebook is clearly labeled to reflect its role in the pipeline (e.g., preprocessing, model development, evaluation). This is where the majority of experimentation and results documentation is performed. 

The notebooks are run in order specified below for reproduceability:
   1. `1_preprocessing_datasetprep.ipynb`
   2. `2_EDA.ipynb`
   3. `3_prelim_tests.ipynb`
   4. `4_model_development_goal1_part1.ipynb`
   5. `5_model_development_goal1_part2.ipynb`
   6. `6_model_development_goal2.ipynb`
   7. `7_model_development_goal3.ipynb`

### `scripts/`
Python scripts containing reusable functions and modular code for preprocessing.
- `helpers.py`: small helper functions for preprocessing

### `models/`
Stores serialized model objects (e.g., `.pkl`, `.h5`, `.pt`) for reuse and deployment. 

### `saved_dfs/`
Processed and intermediate DataFrames saved during preprocessing, transformation, and feature engineering steps. Useful for debugging, version control, and avoiding redundant computation.
Please click link to access: https://drive.google.com/drive/folders/1TjwGqS5w_IMuMZcWZ8XfE-2gePYrm1wn?usp=sharing
   1. `df_for_models.csv`- Finalized dataset for modelling
   2. `merged_df_dict.csv` - Datatypes dictionary
   3. `merged_df_v1.csv` - Checkpoint saved for large dataset manipulation


### `datasets/`
Contains raw and cleaned datasets used in this project. Large files are either compressed or linked via scripts to download from source (e.g., Kaggle). Please click link to access: https://drive.google.com/drive/folders/1vTWHbv71GVw2dGre7YmasxUPrUpfiMYP?usp=sharing 

   1. `raw_partner_headlines.csv` - Dataset from Kaggle chosen for this project  
   2. `nasdaq_screener_1742264403037.csv` - Financial Dataset from NASDAQ
   3. `analyst_ratings_processed.csv` - Dataset from Kaggle (not chosen for this project)
   4. `raw_analyst_rating.csv` -  Dataset from Kaggle (not chosen for this project)



## Literature Review

Recent advancements in Natural Language Processing (NLP) and machine learning (ML) have opened new pathways for stock market forecasting, trading automation, and investor support systems. This review synthesizes key research aligned with our three project goals:

1. Predicting Buy/Hold/Sell for each stock
2. Recommending stocks based on investor profiles
3. Building an NLP-based trading bot

#### *1. Predicting Buy/Hold/Sell Signals Using News Sentiment (Goal 1)*

Several studies validate the role of news sentiment in predicting short-term stock movements. Joshi et al. (n.d.) developed a sentiment-driven classification model using Apple Inc. news articles. By applying Naïve Bayes, Random Forest, and SVM algorithms to tf-idf representations of news data, they achieved over 80% accuracy in classifying news polarity and showed clear correlation with stock price movements. This directly supports the feasibility of Buy/Hold/Sell signal generation through sentiment-aware classifiers. 

Hossain et al. (2021) benchmarked seven machine learning and two deep learning models on 3,383 global news headlines. Bernoulli Naïve Bayes outperformed other algorithms with 82.68% accuracy, highlighting the predictive power of contextual headline sentiment. Their results show the importance of headline-level sentiment, an asset for time-sensitive trade signals. To improve accuracy, noise reduction methods were implemented by only considering JJ (Adjective), VB (Verb in base form), and RB (Adverb) Parts of Speech tag from Penn Treebank annotation .

Maqbool et al. (2023) introduced an MLP Regressor model combining sentiment scores (VADER, TextBlob, and Flair) with historical stock data. This hybrid method achieved up to 90% accuracy for 10-day stock trend predictions, demonstrating how sentiment features can enhance short-term financial forecasts when combined with numerical features.

Shah et al. (2018) took a domain-specific approach using a financial sentiment dictionary tailored for the pharmaceutical sector. Their model yielded 70.59% directional accuracy in short-term predictions, offering a rules-based alternative to supervised learning approaches.

#### *2. Recommending Stocks Based on Investor Profiles (Goal 2)*

While personalized recommendation systems are not the main focus of the reviewed papers, several findings contribute building blocks toward this goal. Hossain et al. (2021) categorized headlines by topics such as politics, business, and health, offering a way to align stock suggestions with investor interests and sector preferences. The evaluation methods for the models focus on accuracy, learning rate, and classifier performance metrics such as receiver operating characteristics (ROC).

Similarly, Shah et al. (2018) and Maqbool et al. (2023) demonstrated that sector-specific sentiment—such as pharmaceuticals or automobiles—can influence the predictability of stock trends. This indicates the potential to recommend stocks tailored to an investor’s time horizon or risk appetite by combining sentiment-driven sector signals with investor profiles.

Furthermore, Maqbool et al.’s model, which evaluated prediction accuracy across different companies, reveals which sectors or stocks are more sensitive to sentiment-based models—insight that can be used to fine-tune recommendations by investment style or volatility tolerance.

#### *3. Building an NLP-Based Trading Bot (Goal 3)*

The selected literature provides a robust foundation for developing an **NLP-based trading bot**. Core components such as real-time news ingestion, sentiment extraction, and prediction modeling are extensively addressed.

Shah et al. (2018) implemented a rule-based NLP model that calculates sentiment from financial news using a custom dictionary, then maps sentiment scores to actionable decisions (Buy, Sell, Hold). This logic forms a natural decision layer in a trading bot pipeline.

Joshi et al. (n.d.) and Hossain et al. (2021) contribute to the classification engine of such a bot, demonstrating the use of TF-IDF and Bag-of-Words representations, along with supervised classifiers (e.g., SVM, Bernoulli NB) for real-time sentiment inference.

Maqbool et al. (2023) further enhances bot architecture by integrating multiple sentiment analyzers with a deep learning-based regressor (MLP). Their experimental pipeline predicts short-term trends using both news and price data—ideal for an intelligent agent that adapts to market sentiment dynamically.

Together, these works form an end-to-end NLP trading bot framework—from scraping, preprocessing, and sentiment scoring, to predictive modeling and decision execution.

Collectively, these studies provide strong evidence that NLP-based sentiment analysis, when combined with traditional ML techniques, enables robust stock classification, personalized recommendations, and intelligent trading systems. They show that:

- **Buy/Hold/Sell predictions** can be enhanced with contextual news sentiment,
- **Investor-tailored stock suggestions** can be informed by sectoral sentiment trends, and
- **NLP-based trading bots** are technically feasible by integrating sentiment scoring, classification, and decision rules into an automated framework.

## Benchmarking

**Goal 1: Predict Buy/Hold/Sell (Classification of News Headlines)**

| **Model**                   | **Accuracy** | **Computational Efficiency** | **Scalability**            | **Interpretability**              | **Real-Time Suitability** |
| --------------------------- | ------------ | ---------------------------- | -------------------------- | --------------------------------- | ------------------------- |
| Logistic Regression         | Good         | Fast                         | High                       | High                              | High                      |
| SVM                         | High         | Moderate to slow             | Low to moderate            | Moderate                          | Low                       |
| Random Forest               | High         | Moderate                     | Moderate                   | Moderate (via feature importance) | Moderate                  |
| Gradient Boosting (XGBoost) | Very high    | Moderate to high             | Moderate (requires tuning) | Low to moderate                   | Moderate                  |
| MLP                         | Very high    | Medium                       | Medium                     | Moderate                          | Medium                    |
| LSTM                        | Excellent    | Low (slow without GPU)       | Low                        | Low                               | Low                       |
| Naive Bayes                 | Good         | Fast                         | High                       | High                              | High                      |


**Goal 2: Recommend Stocks Based on Investor Profile**

| **Model**                   | **Accuracy / Match Rate** | **Computational Efficiency** | **Scalability** | **Interpretability** | **Integration with Profile Features** |
| --------------------------- | ------------------------- | ---------------------------- | --------------- | -------------------- | ------------------------------------- |
| Logistic Regression         | Good                      | Fast                         | High            | High                 | Moderate                              |
| Random Forest               | High                      | Moderate                     | Moderate        | Moderate             | High                                  |
| Gradient Boosting (XGBoost) | Very high                 | Moderate to high             | Moderate        | Low to moderate      | High                                  |
| Rule-Based Filtering        | Varies; profile-aligned   | Very fast                    | Excellent       | Very high            | Very high                             |

**Goal 3: Build an NLP-Based Trading Bot**

| **Model / System**  | **Accuracy** | **Computational Efficiency** | **Scalability** | **Real-Time Suitability** | **Ease of Integration** | **Interpretability** |
| ------------------- | ------------ | ---------------------------- | --------------- | ------------------------- | ----------------------- | -------------------- |
| LSTM-Based Bot      | High         | Slow (GPU preferred)         | Limited         | Moderate to low           | Complex                 | Low                  |
| BERT Fine-tuned     | Very high    | Very slow                    | Low             | Low                       | Complex                 | Low                  |
| TF-IDF + Classifier | Good         | Fast                         | High            | High                      | Easy                    | High                 |
| Rule-Based Logic    | Varies       | Instantaneous                | Excellent       | High                      | Very easy               | Very high            |



## Preliminary Experiments

Before full-scale implementation, we conducted a series of preliminary experiments to test the feasibility of our modeling strategy using financial news headlines, investor profiles, and structured stock metadata. These experiments helped identify suitable model types, validate preprocessing pipelines, and confirm that sentiment-enhanced features could drive actionable predictions.

### Objectives

- Explore classification and recommendation models using structured and text features
- Evaluate deep learning–based collaborative filtering for investor-stock matching
- Assess compatibility and performance of modeling pipelines in Google Colab
- Guide selection of models and feature representations for further development

### Models and Techniques Tested

**Supervised Learning (Classification):**

- **Random Forest** with `GridSearchCV`, achieving macro F1-score ~0.94
- **SVM (Support Vector Machine)** with RBF kernel and cross-validation
- **Logistic Regression** integrated into a preprocessing pipeline
- **MLP (Multi-Layer Perceptron)** with `Keras`, using TF-IDF and structured inputs
- **LSTM** sequence model with `Embedding`, padding, and dense classification head
- **Naive Bayes** with TF-IDF features for text classification, emphasizes simplicity and speed for baseline model evaluation.

**Unsupervised and Recommendation Methods:**

- **KMeans Clustering** applied to stock metadata and sentiment for grouping
- **Content-Based Filtering** via TF-IDF vectors and cosine similarity
- **Deep Learning–Based Collaborative Filtering**:
  - Implemented using `Keras` with embedding layers for investor and stock IDs
  - Interaction score predicted via dense layers and a sigmoid output
  - Modeled the likelihood of investor-stock alignment based on profile embeddings

```python
# Deep Collaborative Filtering Summary
embedding_investor = Embedding(input_dim=100, output_dim=10)(input_investor)
embedding_stock = Embedding(input_dim=1000, output_dim=10)(input_stock)
concat = Concatenate()([Flatten()(embedding_investor), Flatten()(embedding_stock)])
output = Dense(1, activation="sigmoid")(Dense(128, activation="relu")(concat))
```

### Preprocessing

- Text cleaning, tokenization, and lemmatization via **spaCy**
- Sentiment scoring using **VADER**
- Feature extraction: TF-IDF, headline length, market cap category, etc.
- Label encoding for structured categorical attributes
- Sequence preparation (tokenizer and padding) for deep models

### Key Findings

- **Random Forest** was the best-performing classical model, with high F1 and interpretability.
- **LSTM and MLP models** successfully captured headline semantics when paired with structured inputs.
- **Deep collaborative filtering** produced promising results, demonstrating the ability to recommend stocks based on learned investor-stock relationships.
- **KMeans** helped reveal investor segments and clustering potential.
- All pipelines ran successfully in **Google Colab**, with GPU acceleration leveraged for deep learning models.

### Conclusion

The preliminary experiments validated both the technical feasibility and predictive potential of our multi-model approach. From classical classifiers to deep recommendation models, each method contributed valuable insights for building a scalable, sentiment-aware, investor-personalized stock trading system.

# Model Implementation
---

## Framework Selection

This project uses a combination of NLP, machine learning, and deep learning frameworks selected based on model complexity, task requirements, and team familiarity.

- **Text Preprocessing**:
  - `spaCy` for efficient tokenization and lemmatization
  - `NLTK` with `VADER` for sentiment scoring of headlines
- **Classical Machine Learning**:
  - `scikit-learn` for implementing Logistic Regression, Random Forest, SVM, KMeans, and similarity-based recommenders
- **Deep Learning**:
  - `TensorFlow` and `Keras` for MLPs and LSTM models used in classification and bot logic
- **Advanced NLP**:
  - `Hugging Face Transformers` for fine-tuning BERT on financial headline classification
- **Reinforcement Learning**:
  - `PyTorch` for implementing and training custom RL agents in the trading bot module

These frameworks enabled a modular pipeline across all project goals, balancing performance, interpretability, and development speed.

## Dataset Preparation

The dataset preparation stage focused on transforming raw financial news data into a structured, analyzable format suitable for downstream NLP and machine learning tasks across all project goals. This included parsing headlines, enriching with metadata, performing text preprocessing, generating sentiment labels, and engineering features for model input.

### **1. Data Loading and Integration**

- Combined headline data with stock metadata, including `Market Cap`, `Country`, `IPO Year`, `Sector`, and `Industry`.
- Loaded additional fields such as `Publisher`, `Date`, `Stock Ticker`, and `Recommendation` labels (Buy/Hold/Sell).

### **2. Text Normalization and Tokenization**

- Tokenized each headline using word tokenizers.
- Applied **lowercasing**, **punctuation removal**, and **stopword filtering**.
- Generated normalized token lists, filtered tokens, and lemmatized tokens for each headline.

### **3. Sentiment Labeling**

- Used the **VADER Sentiment Analyzer** to compute sentiment scores for each headline.
- Created sentiment labels based on score thresholds:
  - **Positive**: sentiment score > 0.5
  - **Negative**: sentiment score < -0.5
  - **Neutral**: otherwise

### **4. Feature Engineering**

- Extracted headline-level features:
  - **Length of headline**
  - **Word count**
  - **Day of week**, **month**, and **year**
- Created derived features like `Market_Cap_Category` (Small, Mid, Large) based on numerical thresholds.
- Engineered NLP features: `tokens`, `normalized_tokens`, `filtered_tokens`, `lemmas`.

### **5. Label and Feature Encoding**

- Encoded target labels (`recommendation`, `sentiment_label`) for classification.
- Encoded categorical features such as:
  - `Publisher`
  - `Country`
  - `Sector`
  - `Industry`
- Applied `LabelEncoder` and one-hot encoding where necessary.

### **6. Final Dataset Structure**

Each row in the final processed dataset contains:

- Cleaned and tokenized headline
- Sentiment score and label
- Stock fundamentals
- Encoded categorical variables
- Engineered NLP and numerical features
- Ground truth recommendation label (Buy, Hold, Sell)

This prepared dataset serves as the foundation for:

- **Goal 1**: Training models to predict Buy/Hold/Sell signals.
- **Goal 2**: Stock recommendation aligned to investor profiles.
- **Goal 3**: Automated decision-making in the NLP-based trading bot.



## Model Development

The model development process was modular and aligned with each project goal, ensuring reusability, interpretability, and ease of experimentation. All code was implemented in Google Colab using structured notebooks, with consistent preprocessing pipelines and clear separation of logic for each task.

### Goal 1: Predict Buy / Hold / Sell (Classification)

We implemented and benchmarked the following supervised learning models:

- **Logistic Regression**, **Random Forest**, **XGBoost**, and **SVM** using `scikit-learn`
- **Multi-Layer Perceptron (MLP)** using `TensorFlow/Keras`
- **LSTM/BiLSTM** architectures for sequence modeling
- **BERT + MLP** for deep contextual headline classification using `Hugging Face Transformers`

Each model was trained on engineered features including lemmatized text, sentiment score, headline metadata, and stock attributes.





### Goal 2: Recommend Stocks Based on Investor Profile

We designed modular recommender systems that match user profiles (e.g., investment style, risk tolerance) with appropriate stocks:

- **Content-Based Filtering** using cosine similarity on stock metadata
- **KNN-Based Recommenders** for user-stock similarity
- **KMeans Clustering** to segment stocks into investor-friendly categories
- **Collaborative Filtering** using rating-like signals





### Goal 3: NLP-Based Trading Bot

A prototype trading bot was developed to simulate and automate trading actions based on real-time news sentiment and model predictions:

- **ML-Based Decisions** via XGBoost and MLP
- **Temporal Modeling** with LSTM to handle news flow sequences
- **Reinforcement Learning (Q-Learning)** to train a decision agent that maximizes cumulative profit based on simulated feedback



## Training and Fine-tuning

After selecting appropriate models for classification, recommendation, and trading automation, we performed rigorous training and hyperparameter tuning to maximize performance and generalization.

### 1. Traditional Machine Learning Models

We applied systematic hyperparameter optimization for all classical classification algorithms:

- **Logistic Regression**
  Tuned `C` (inverse of regularization strength) in the range `[0.01, 0.1, 1, 10]` to balance bias-variance. Integrated with pipeline and cross-validation.
- **Random Forest**
  Explored `n_estimators` from 100 to 300, `max_depth` values up to 10, and `min_samples_split` between 2–10. Grid search was applied with 5-fold cross-validation.
- **XGBoost**
  Fine-tuned learning rate (`eta`), `max_depth`, `subsample`, and `lambda` parameters. Early stopping based on validation F1-score was used to avoid overfitting.
- **Support Vector Machine (SVM)**
  Tuned `C` and `gamma` for RBF kernel; also compared performance with linear kernel. Emphasis was placed on margin control and generalization.
  **Naive Bayes**
- Used TF-IDF features for text classification, emphasizes simplicity and speed for baseline model evaluation.

> All traditional models were evaluated using `StratifiedKFold` cross-validation and optimized for macro-averaged F1-score due to class imbalance in Buy/Hold/Sell labels.

### 2. Deep Learning Models

We used Keras (TensorFlow backend) to develop and tune neural architectures:

- **MLP (Multi-Layer Perceptron)**
  Layer architecture: 2–3 dense layers with ReLU activation, dropout layers (`rate=0.3`), and batch normalization. Output layer used softmax for 3-class classification. Hyperparameters included:

  - Learning rate (Adam optimizer)
  - Dropout rate
  - Hidden layer units (64–128)

- **LSTM / BiLSTM**
  Used preprocessed sequences with padding and embedding layers. Hyperparameters included:

  - `max_sequence_length`
  - `embedding_dim`
  - `hidden_units` (64/128)
  - Recurrent dropout to avoid overfitting

  Early stopping and `ReduceLROnPlateau` callbacks were used to stabilize training.

- **BERT Fine-Tuning**
  Pretrained BERT (from Hugging Face `transformers`) was fine-tuned on tokenized headlines using:

  - Learning rate scheduling (`get_linear_schedule_with_warmup`)
  - Layer-wise learning rate decay
  - AdamW optimizer with weight decay
  - Custom classification head for 3-class output

  Training leveraged GPU acceleration in Google Colab for faster iteration.

### 3. Recommendation Models

We explored a variety of recommendation methods based on metadata and user-stock interactions:

- **Content-Based Filtering**
  Used cosine similarity on TF-IDF headline vectors. Tuned minimum similarity threshold for match inclusion.
- **Clustering (KMeans)**
  Tuned number of clusters (`k=3–5`) based on investor metadata, sentiment, and market cap. Silhouette score was used for validation.
- **Collaborative Filtering (Deep Learning)**
  A neural network was constructed using embedding layers for investor and stock IDs. The embedded vectors were concatenated and passed through dense layers to predict interaction likelihood.
  The model was trained using binary crossentropy loss with early stopping and batch tuning.

All models underwent careful parameter tuning to balance bias, variance, and training efficiency. Transfer learning (via BERT) was leveraged for semantic representation, while classical and deep models were fine-tuned through iterative experimentation. Colab GPU runtime was utilized to accelerate deep learning workflows.



## Evaluation and Metrics

To ensure robust, generalizable performance across all models, we applied task-specific evaluation strategies aligned with each project goal. These strategies combined domain-relevant metrics with best practices in validation and optimization to ensure both accuracy and deployment-readiness.

### 1. Classification Models  

**(Goal 1: Predict Buy / Hold / Sell based on News Headlines)**

For all classification models—including Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost, MLP, LSTM, and BERT—we used the **F1-macro score** as the primary evaluation metric. This allowed us to fairly assess performance across imbalanced classes (Buy, Hold, Sell).

**Additional metrics included:**

- Balanced accuracy for overall classification reliability  
- Precision and recall per class, derived from detailed classification reports  

**Validation and tuning strategies:**

- `RandomizedSearchCV` with stratified sampling for balanced cross-validation  
- Manual validation sets for efficient iteration  
- Progressive sampling to evaluate model behavior on increasing data sizes  

This approach ensured that model performance was both statistically sound and practically applicable.

### 2. Recommendation Models  

**(Goal 2: Recommend Stocks Based on Investor Profile)**

Our recommendation models—content-based, KNN, clustering, and deep collaborative filtering—were evaluated not by traditional accuracy metrics, but by how well the recommendations aligned with individual investor profiles.

**Key evaluation strategies included:**

- Cosine similarity to measure semantic closeness between stock features and investor preferences  
- Match rate to quantify how many recommendations aligned with profile attributes (e.g., risk tolerance, market cap)  
- Silhouette score to evaluate clustering quality in unsupervised models  
- Profile alignment checks, combining heuristic logic and manual inspection  

For collaborative filtering, we observed how well embedding layers for investors and stocks converged to meaningful representations that generalized well to unseen pairs.

### 3. Trading Bot Models  

**(Goal 3: Build an NLP-Based Trading Bot)**

For models integrated into the trading bot (LSTM, BERT, and rule-based pipelines), we used evaluation methods suited for real-time, trend-sensitive tasks.

**Primary metrics included:**

- Root Mean Square Error (RMSE) on inverse-transformed (non-scaled) stock price predictions  
- Mean Absolute Error (MAE) for robustness to outliers  
- Directional accuracy to evaluate the model’s ability to predict upward or downward movements  

To preserve temporal integrity during validation, we used **forward-chaining time series splits**, preventing future data leakage.

**Resource and performance optimizations included:**

- Batch prediction to handle large headline inputs efficiently  
- GPU acceleration for BERT and LSTM in Google Colab  
- Caching of preprocessed features for faster repeated evaluations  
- Early stopping for deep learning models, guided by validation loss curves  

These techniques ensured that our models were both accurate and practical for real-time deployment in a trading environment.

Each goal was paired with its own metric-driven evaluation approach:

- **Goal 1** prioritized classification fairness and predictive balance  
- **Goal 2** emphasized personalization and semantic relevance  
- **Goal 3** focused on predictive precision over time and runtime efficiency  

Together, these metrics provided a comprehensive foundation for refining our models and guiding development decisions throughout the project.

# References (APA Style)
---

1. Hossain, S. S., Arafat, Y., & Hossain, M. E. (2021). *Context-based news headlines analysis: A comparative study of machine learning and deep learning algorithms*. *Vietnam Journal of Computer Science*, 8(4), 513–527. https://doi.org/10.1142/S2196888822500014
2. Joshi, K., Bharathi, H. N., & Rao, J. (n.d.). *Stock trend prediction using news sentiment analysis*. K. J. Somaiya College of Engineering. 
3. Maqbool, J., Aggarwal, P., Kaur, R., Mittal, A., & Ganaie, I. A. (2023). *Stock prediction by integrating sentiment scores of financial news and MLP-Regressor: A machine learning approach*. *Procedia Computer Science*, 218, 1067–1078. https://doi.org/10.1016/j.procs.2023.01.086
4. Shah, D., Isah, H., & Zulkernine, F. (2018). *Predicting the effects of news sentiments on the stock market*. In *2018 IEEE International Conference on Big Data (Big Data)* (pp. 4705–4710). IEEE. https://doi.org/10.1109/BigData.2018.8622620
