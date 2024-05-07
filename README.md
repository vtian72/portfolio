# Welcome! 

Hey there! I'm Vincent, a Data Scientist in Cambridge, MA but originally from Perth, Australia 🇦🇺

I'm currently...

- Completing my Master's in Business Analytics (MBAn) @ MIT graduating in August 2024
- Working on my capstone project @ Macy's to build customer price range models and recommendation systems
- Wrapping up my time as VP @ Sloan Blockchain Club (SBC) 🥲

These were some events I helped organize at SBC

- Web3 Entrepreneurship with Avalanche's Global Accelerator - Feb 14, 2024
- AMA: Bitcoin Pizza Party 🍕 - Nov 21, 2023
- MIT x Harvard Business School x Harvard Law School Mixer - Oct 22, 2023

## Experience 

**Data Scientist Intern @ Macy's (_Feb 2024 - Present_)**
- MIT Capstone 👕
- Building customer price range models and recommendation systems 
- Python, SQL, BigQuery, VertexAI

**Data Scientist Intern @ Handle Global (_Sep 2023 - Dec 2023_)**
- MIT Analytics Lab 🏥
- Comparing medical equipment prices to inflation and forecasting future prices 
- Python, ARIMA, Prophet

**Data Scientist @ Quantium (_Feb 2021 - June 2023_)**
- Health & Government 🧑‍⚕️, Product Analytics ⌚
- Population forecasting, churn predictions, end-to-end dynamic dashboards, automated decision support tool for state government, and many more ...
- Python (XGBoost, Dash Plotly), R, Git, SQL, Snowflake, Azure Data Factory, PowerBI

**Data Scientist Intern @ Fortescue Metals Group (_Nov 2020 - Feb 2021_)**
- Machine Learning & AI Team 🛢️
- Predicting oil quality from sample oil testing data
- Python (scikit-learn, LightGBM, XGBoost, fast.ai), AWS Sagemaker, AWS MLOps

# Check out some of my projects!

### AI-Driven Chat Sentiment Insights (Work in Progress 🔨)

Currently leading a team to develop the first end-to-end AI-powered analytics tool that scrapes text data from Telegram, Discord (TBD) and Twitter (TBD) to generate personalized market and sentiment analysis insights using LLMs. 

In our product workflow, when a user links their social media accounts, we automatically gather all their message data. Subsequently, we employ conventional techniques like Named Entity Recognition (NER) and TF-IDF to extract valuable features. These features are then transformed into embeddings, which are used to cluster related messages. Finally, we feed this clustered data into a Language Model (LM) such as GPT-3, GPT-4, or Gemini to generate the final outputs. Looking ahead, we plan to refine our LM model further using Reinforcement Learning from Human Feedback (RLHF), by presenting it with examples of effective and useful summaries.

We will be using AWS (S3, DynamoDB) to store our data.

We are still working on the initial product but here is a sneak peek of what we have built so far!

![Dashboard](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/tool.png)

### Is It Possible to Imitate Shakespeare with LLMs and Style Transformers? 🎭

<u>Goal</u>

The primary goal of the project is to determine how effectively generative AI models can replicate the distinctive writing style of William Shakespeare. This involves developing a quantitative benchmark to measure the fidelity of these models in capturing the depth, style, and subtleties of Shakespearean English. The project aims to transition from qualitative assessments to a structured, automatic, and quantifiable evaluation methodology, enhancing the capabilities of AI in reproducing complex literary styles.

<u>Finetuning</u>

GPT-2 and GPT DaVinci were finetuned on 1k/48k rows of data from the Shakescleare dataset, consisting of modern English passages alongside their Shakespearean translation. We also utilized the output by the Style Transformer model (Reformulating Unsupervised Style Transfer as Paraphrase Generation, by Kalpesh Krishna, John Wieting, Mohit Iyyer).

To fine-tune GPT-2 a specialized formatting was adopted. Each row of data was transformed into the following:

<s> (start token) + English translation + </s> (end token) + >>>> + <p> (start token for Shakespeare translation) + Shakespearean translation + </p> (end token for Shakespeare translation)

Since DaVinci as inherest prompt-completion functionality, we structured the dataset as:

prompt: System role: You are an expert author on Shakespeare. Write the following quote like how Shakespeare would say it: + English translation

completion: This is how Shakespeare would say it: + Shakespearean translation

<u>Results</u>

| Model Output        | Style Classifier | BLEU Score | Rouge-N Score | Cosine Similarity | Jaccard Similarity | PINC Score |
|---------------------|------------------|------------|---------------|-------------------|--------------------|------------|
| GPT-2               | 53.21%           | 6.74%      | 14.52%        | 83.29%            | 19.76%             | 85.61%     |
| GPT-DaVinci         | 79.69%           | 5.59%      | 19.59%        | 87.74%            | 23.56%             | 80.88%     |
| Style Transformer A | 28.66%           | 7.21%      | 22.91%        | 93.45%            | 36.38%             | 76.66%     |
| Style Transformer B | 34.23%           | 7.32%      | 20.91%        | 93.30%            | 34.40%             | 78.20%     |


[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Shakespeare%20Style.pdf)

### Get Real: Real vs Fake Image Detection 🪪
[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Project%20Report%20-%20Fake%20Product%20Scam%20Detector.pdf)

### Strategic Microchip Supply Chain Route Optimization 🚢
[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Enhancing_Efficiency_in_Microchip_Distribution__Strategic_Supply_Chain_Route_Optimization.pdf)

### Fetal Health Classification Using Cardiotocography Data 👶
[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Fetal%20Health%20Classification.pdf)




