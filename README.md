# Welcome! 

Hey there! I'm Vincent, a Data Scientist with ~3 years of experience based in Cambridge, MA, but originally from Perth, Australia 🇦🇺

I'm currently...

- Completing my Master's in Business Analytics (MBAn) @ MIT graduating in August 2024
- Working on my capstone project @ Macy's to build customer price range models and recommendation systems
- Wrapping up my time as VP @ Sloan Blockchain Club (SBC) 🥲

## Experience 

**Data Scientist @ Macy's (_Feb 2024 - Present_)**
- MIT Capstone 👕
- Building customer price range models and recommendation systems 
- Python, SQL, BigQuery, VertexAI

**Data Scientist @ Handle Global (_Sep 2023 - Dec 2023_)**
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

## AI-Driven Chat Sentiment Insights (Work in Progress 🔨)

Currently leading a team to develop the first end-to-end AI-powered analytics tool that scrapes text data from Telegram, Discord (TBD) and Twitter (TBD) to generate personalized market and sentiment analysis insights using LLMs. 

In our product workflow, when a user links their social media accounts, we automatically gather all their message data. Subsequently, we employ conventional techniques like **Named Entity Recognition (NER) and TF-IDF** to extract valuable features. These features are then transformed into embeddings, which are used to cluster related messages. Finally, we feed this clustered data into a **Language Model (LM) such as GPT-3, GPT-4, or Gemini** to generate the final outputs. Looking ahead, we plan to **refine our LM model further using Reinforcement Learning from Human Feedback (RLHF)**, by presenting it with examples of effective and useful summaries.

We will be using AWS (S3, DynamoDB) to store our data.

We are still working on the initial product but here is a sneak peek of what we have built so far!

**Tools: Python, AWS (S3, DynamoDB), LLMs (GPT-3, GPT-4, Gemini)**

![Dashboard](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/tool.png)

## Get Real: Real vs Fake Image Detection 🪪

<u>Goal</u>

Today, GenAI has the capability to produce realistic product images swiftly, significantly accelerating the process of creating scam websites and enhancing social engineering strategies employed by scammers. Our project aims to develop a customized model that can classify product images as real or AI-generated with a focus on e-commerce product images that can be used for scams. Potentially, our product can be combined with other scam detection tools, e.g. scanning the registered domain owner, the activity of the IP address etc, to design an effective fake product scam detector. 

![Dashboard](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/ai_real_images.png)

**These are all AI-generated images!**

<u>Data</u>

Our dataset comprises 6,000 images. We randomly sampled 3,000 real product images from the Amazon Berkeley Objects dataset and ran it through Google Gemini Pro to generate a one-line image caption of the object. These descriptions are then used to generate a “like-for-like” fake product image dataset comprising 3000 images using DALL-E 2, to create balanced classes for training our dataset. In creating our “real” product dataset, we made the effort to select a variety of product types against varying backgrounds. This is so that we are able to train a model on the diverse range of images a GenAI model can produce.

![Dashboard](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/ai_real_process.png)

<u>Model Training</u>

We applied transfer learning on the following models with only the output layer trained. 

1. **VGG-19**, known for its simplicity and depth
2. **ResNet50**, alleviates the vanishing gradient problem, allowing for the training for very deep networks
3. **EfficientNet** capable of generalizing well on unseen data.

<u>Results</u>

The following models were trained using the A-100 GPU on Google Colab.

| Model            | Accuracy | False (-)ve Rate | Training Time (s) |
|------------------|----------|------------------|-------------------|
| VGG-19           | 95.11%   | 6.62%            | 245               |
| ResNet50         | 99.00%   | 0.45%            | 270               |
| EfficientNetB0   | 99.22%   | 0.45%            | 293               |
| EfficientNetB3   | 99.00%   | 0.89%            | 515               |
| EfficientNetB7   | 99.44%   | 0.88%            | 1431              |

The following show the confusion matrix on 900 samples of testing data.

True Negative (TN): Model correctly identifies images as fake
True Positive (TP): Model correctly identifies image as real
False Negative (FN): Model fails to flag out an image as fake
False Positive (FP): Model erroneously flags out an image as false when it is actually real


<u>Figure 3.1: Confusion Matrix for Fine-tuned VGG 19 model over 20 epochs</u>

|                 | **Predicted Labels**   |                    |
|-----------------|-----------------------|--------------------|
| **Actual Labels** | **Fake**              | **Real**           |
| **Fake**          | TN: 419   | FN: 31|
| **Real**         | FP: 13   | TP: 437|

The results show that VGG-19 may not be the best at discerning the false images from the real images, as the number of False Negatives are still quite high. 

<u>Figure 3.2: Confusion Matrix for Fine-tuned ResNet50 model over 20 epochs</u>

|                 | **Predicted Labels**   |                    |
|-----------------|-----------------------|--------------------|
| **Actual Labels** | **Fake**              | **Real**           |
| **Fake**          | TN: 448   | FN: 2|
| **Real**         | FP: 7   | TP: 443|

In comparison, ResNet50 was able to improve drastically compared to VGG19 in the number of False Positives and False Negatives, i.e we are now able to flag out fake images as image and real images as real better.

<u>Performance on newer GenAI models</u>

To mimic real-world conditions and test our hypothesis on whether the model performs better on older GenAI models, we also created a variety of GenAI images on newer GenAI models like DALL-E 3, which are unseen to our model. Expectedly, our model performs poorly on the DALL-E 3 dataset. We compared the fine-tuned models fine-tuned over 20 epochs for consistency. EfficientNetB0 performs best on the unseen GenAI data, due to its advanced architecture features like the squeeze-and-excitation optimization and MBConv blocks that allow it to learn both coarse and fine features from the data and make them more adaptable to different types of images. 

Of the 129 DALL-E 3 images tested, the following results were derived:
- VGG-19 model: 24.31% accuracy (31 correct; 98 incorrectly predicted as real)
- ResNet50 model: 48.43% accuracy (63 correct; 66 incorrectly predicted as real)
- EfficientNetB0: 50.39% accuracy (65 correct; 64 incorrectly predicted as real)

This is testament to DALL-E 3’s ability to generate convincing images, but also highlights the challenges in updating detection algorithms to keep up with the advancements in GenAI image generation.

<u>GetReal Chrome Extension!</u>

We also embedded the models into a Chrome Extension which can be used to determine if images online are real or fake!

![GIF](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/getreal.gif)

**Tools: Python (Keras, TensorFlow), GenAI (Gemini, DALL-E 2, DALL-E 3)**

[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Project%20Report%20-%20Fake%20Product%20Scam%20Detector.pdf)


## Is It Possible to Imitate Shakespeare with LLMs and Style Transformers? 🎭

<u>Goal</u>

The objective is to assess the ability of generative AI models to emulate William Shakespeare's unique writing style accurately. This entails comparing the performance of different models using a range of metrics designed to measure stylistic accuracy, including BLEU, Rouge-N, Cosine Similarity, Jaccard Similarity, and PINC Score.

<u>Finetuning</u>

GPT-2 and GPT DaVinci were finetuned on 1k/48k rows of data from the Shakescleare dataset, consisting of modern English passages alongside their Shakespearean translation. We also utilized the output by the Style Transformer model (Reformulating Unsupervised Style Transfer as Paraphrase Generation, by Kalpesh Krishna, John Wieting, Mohit Iyyer).

To **fine-tune GPT-2 a specialized formatting was adopted**. Each row of data was transformed into the following:

\<s> (start token) + English translation + \</s> (end token) + >>>> + \<p> (start token for Shakespeare translation) + Shakespearean translation + \</p> (end token for Shakespeare translation)

Since **DaVinci has inherent prompt-completion functionality**, we structured the dataset as:

prompt: System role: You are an expert author on Shakespeare. Write the following quote like how Shakespeare would say it: + English translation

completion: This is how Shakespeare would say it: + Shakespearean translation

<u>Results</u>

ST: Style Transformer

| Metric            | GPT-2  | GPT-DaVinci | ST-A  | ST-B  |
|-------------------|--------|-------------|-------|-------|
| Style Classifier  | 53.21% | 79.69%      | 28.66%| 34.23%|
| BLEU Score        | 6.74%  | 5.59%       | 7.21% | 7.32% |
| Rouge-N Score     | 14.52% | 19.59%      | 22.91%| 20.91%|
| Cosine Similarity | 83.29% | 87.74%      | 93.45%| 93.30%|
| Jaccard Similarity| 19.76% | 23.56%      | 36.38%| 34.40%|
| PINC Score        | 85.61% | 80.88%      | 76.66%| 78.20%|

Example 1


|----------------|-------------------------------------------------------------|
| English        | My generosity to you is limitless as the sea, and my love is as deep |
| Shakespeare    | My bounty is boundless as the sea, My love as deep           |
| GPT-2          | My goodness to thee is as infinite as the sea, And my love as deep |
| GPT-DaVinci    | My love to you is sound, sans crack or flaw                  |
| ST-A           | I’m boundless, love is deep                                  |
| ST-B           | I’m boundless, love is deep                                  |


Example 2

|----------------|-------------------------------------------------------------|
| English        | I’ll frown at them as they pass by, and they can react however they want |
| Shakespeare    | I will frown as I pass by, and let them take it as they list       |
| GPT-2          | I’ll frown at them as they pass by, And they may be as they will   |
| GPT-DaVinci    | I’ll set upon them as they pass along                              |
| ST-A           | I’ll frown and let them take it as they list it                    |
| ST-B           | I’ll frown and let them take it as a listing                       |


Example 3

|----------------|-------------------------------------------------------------|
| English        | Please, sir, have patience                                  |
| Shakespeare    | I do beseech you, sir, have patience                        |
| GPT-2          | I pray you, sir, patience, I pray you sir, patience, I pray you |
| GPT-DaVinci    | Pray you, sir, have patience                                |
| ST-A           | I beg your, sir, be patient                                 |
| ST-B           | I beg your, sir, be patient                                 |

The results table indicates that no single model outperforms others across all metrics, despite Style Transformer A exhibiting superior performance in 4 out of 6 metrics. Additionally, the examples underscore the diverse capabilities of different models in reproducing Shakespearean English. For example, in the first example, GPT-2 and DaVinci were able to retain the poetic depth of the original text whereas Style Transformers A and B simplified the phrase significantly.

**Tools: Python, LLMs (GPT-2, GPT-DaVinci)**

[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Shakespeare%20Style.pdf)

## Strategic Microchip Supply Chain Route Optimization 🚢

<u>Goal</u>

The objective for this project was to help a global microchip producer design an optimal distribution network that incorporates warehouses, shipping routes, and courier services to create the most economical supply chain possible. We wish to minimize total costs, comprising warehouse operations and transportation expenses while adhering to the constraints of demand, supply and shipping logistics.

<u>Baseline</u>

We developed a baseline model for addressing this problem, known as the Yan-Tian Greedy Algorithm. The primary concept behind this approach involves a systematic iteration through all incoming orders. For each order, we initiate a search through available warehouses and their corresponding freight options, starting from the beginning of the list. The algorithm then assigns the order to the first suitable warehouse-freight pair it encounters, following a thorough evaluation to ensure that all necessary conditions are met before making the assignment. The cost of solution produced by the baseline model is $8,878,241.89.

<u>Formulating the Optimization Problem</u>

The overall objective function that we were minimizing was:

min(Warehouse Cost + Transportation Cost)

Some constraints that we needed to account for:

- each order needs to be assigned to a warehouse
- each order needs to be assigned to a freight assignment
- each warehouse has a daily order capacity
- each product can be stored in some warehouses only
- some warehouses can only service certain customers
- each warehouse can only begin transporting things via some specific warehouse ports
- orders need to be shipped within a certain time
- different parts of the carrier should not exceed a maximum weight

<u>Exploratory Data Analysis</u>

![EDA1](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/warehouse_to_port.png)

This shows that most warehouses are connected to only a single warehouse port and many warehouses are connected to warehouse port 4, which suggests that many warehouses may only be sent via one freight and many orders may be sent through freights going through wwarehouse port 4.

![EDA2](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/cost_perUnit_daily_order_capacity.png)

Most warehouses have negative correlation between cost per unit cost and daily order capacity. We should expect the warehouses with lower cost per unit to have the most number of orders allocated to it.

<u>Results</u>

![Results](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/sankey.png)

The above Sankey shows the optimal warehouse and freight allocation for each of 1000 orders. As from our EDA we can see that the optimal solution includes:
- many orders to pass through warehouse port 4
- many orders allocated to warehouse 3 and warehouse 11 due to their lower daily cost per unit
- not many orders allocated to warehouse 15, 16 or 18 due to their high daily cost.

The final solution resulted in a cost of $5,365,566,57 which is $3,512,675.32 or 39.5% less than the baseline solution.

**Tools: Python, Julia/JuMP(Gurobi)**

[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Enhancing_Efficiency_in_Microchip_Distribution__Strategic_Supply_Chain_Route_Optimization.pdf)

## Other projects you can check out!

### Fetal Health Classification Using Cardiotocography Data 👶

Child and maternal mortality are urgent global concerns. The UN targets ending preventable deaths in newborns and under-5 children by 2030. Cardiotocograms (CTGs) provide vital fetal health data using ultrasound pulses, allowing timely interventions. This cost-effective technology significantly reduces child and maternal mortality, especially in resource-constrained areas. The project aim is to classify fetal health to prevent child and maternal mortality.

![Image](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/fetal_health.png)

**Tools: Python, SMOTE**

[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/Fetal%20Health%20Classification.pdf)

### Enhancing Automotive Customer Segmentation through Comprehensive Imputation Techniques 

An automotive company aims to expand into new markets by introducing its current product lineup (P1, P2, P3, P4, and P5). Extensive market research indicates similarities between the behavior of the new market and the existing one. In their current market, the sales team has categorized customers into four segments (A, B, C, D) and implemented targeted outreach and communication for each segment, resulting in remarkable success. The company intends to apply the same successful strategy to the new markets, where they have identified 2627 potential customers.
Our task at hand is to assist the manager in accurately predicting the appropriate segment for these new customers.

![Image](https://raw.githubusercontent.com/vtian72/portfolio/main/assets/img/ml.png)

**Tools: Python, Julia/JuMP**

[Report](https://github.com/vtian72/portfolio/blob/main/assets/files/ML%20report.pdf)

## Education

- MBAn, Business Analytics @ MIT (Aug 2024)
- BSc, Mathematics and Statistics @ University of Melbourne 
- Cross-Registered, Data Science @ Harvard (Spring '24)
- Study Abroad, Statistics @ University College London (Spring '19)

### Leadership

These were some events I helped organize at SBC

- Web3 Entrepreneurship with Avalanche's Global Accelerator - Feb 14, 2024
- AMA: Bitcoin Pizza Party 🍕 - Nov 21, 2023
- MIT x Harvard Business School x Harvard Law School Mixer - Oct 22, 2023

### List of all the classes I've taken 

**MBAn, Business Analytics @ MIT**

Spring '24
- 15.287 Communicating with Data 
- 15.665 Power and Negotiation
- 15.773 Hands on Deep Learning
- 15.783 Product Development Methods
- 15.819 Marketing and Product Analytics
- ECON2355 Unleashing Novel Data at Scale (Harvard)

Fall '23
- 15.072 Advanced Analytics Edge
- 15.093 Optimization Methods
- 15.095 Machine Learning Under a Modern Optimization Lens
- 15.575 Analytics Lab
- 15.681 From Analytics to Action

**BSc, Mathematics and Statistics @ University of Melbourne**

Sem 2 - 2020
- MAST20018 Discrete Maths and Operations Research
- MAST30028 Numerical Methods & Scientific Computing
- MAST30001 Stochastic Modelling
- MAST30027 Modern Applied Statistics

Sem 1 - 2020
- MAST20009 Vector Calculus
- MAST20026 Real Analysis
- MAST30025 Linear Statistical Models
- COMP30027 Machine Learning
  
Sem 2 - 2019
- COMP20008 Elements of Data Processing
- COMP30020 Declarative Programming
- MAST10007 Linear Algebra
- MAST20005 Statistics

Credit via University of Western Australia
- COMP10001 Foundations of Computing
- COMP10002 Foundations of Algorithms
- COMP20003 Algorithms and Data Structures
- INFO20003 Database Systems
- MAST10006 Calculus 2
- MAST20006 Probability for Statistics

**Study Abroad, Statistics @ University College London**
- MATH0031 Financial Mathematics
- MATH0057 Probability and Statistics
- STAT0007 Stochastic Processes
- STAT0023 Computing for Practical Statistics
