# Project Proposal


## What Is This Project About
This Project is based around building a transformer to generate satirical Onion-like articles based on a text input. 
### Look at MVP 

### Some Stretch Goals 



### Stakeholders 
This will eventually be an internet facing application which will be discoverable by anyone who wants to read what it generates. If time allows I will try to add some capabilities for it to post said article to a social media site, maybe twitter for example. For my MVP though it will be discoverable by the larger internet through its own website. 

## How Does The Project Meet The Requirements
### The Tech Stack 
For the website portions I will be using Flask as a GUI which can interact with the model.

For the model I will be using pytorch and its transformer library. 

I am fairly firmiliar with both as I've used both in the past. On the pytorch side of things there will definitly be a larger learning curve figuring out how to user a transformer effectivly.

### The Data Sources 
#### Static Data Sources 
The first one will be one of the Stanford GloVe datasets, specifically the Wikipedia dataset. This dataset is compossed of preprocessed word embeddings which can be loaded into a transformer without having to train one from scratch. 

I will also be using two datasets I found on Kaggle. Both of these will be used for transfer learning, starting with the imported glove vectors I will specialize it write in the onion's satirical tone and language.
1. The first is a classification dataset called *OnionOrNot*, [link to it here](https://www.kaggle.com/datasets/chrisfilo/onion-or-not). This dataset will be mainly used to define the type of language and topics the transformer should be using. This dataset's shape is roughly 24000 instances of titles and class labels, real titles being labeled 1.
2. The second dataset, which is called *News Summarization*; [link to it here](https://www.kaggle.com/datasets/sbhatti/news-summarization). It is a csv file which contains information on articles from professional news sources like XSum, CNN/Daily Mail, and Multi-News. This dataset's shape is roughly 580000 instances comprised of a record id, content, summary, and news source label. 

#### Timely Data Sources
The goal of this project, one the model has been created is to use the twitter API to ingest data and figure out what trending topics are. Using these topic, I will be generating a fake article with my model which will then be made publically availible to the internet. Also I may attempt posting this article to twitter as either an abreviated version which will fit twitters word limit or just link the full article. This will require using the twitter API.

### Major Analysis and Anticipated Results
Given that this project will be based around getting the best performace out of my model, most of the analysis will be done by seeing how well the model is doing in terms of generatting coherent, sarcastic articles. 

Here is a blog [link](https://angelina-yang.medium.com/how-to-test-nlp-models-from-the-lens-of-software-engineering-3261d22fb8bc) which I will be following that describes how to test NLP models at a basic level.

Here is another post [link](https://medium.com/analytics-vidhya/nlp-transformer-unit-test-95459fefbea9) talking about how to unit test a transformer.

A large part of my testing however will be generating sample data and getting a more qualitative feel for how it's putting sentences together. 



### Deployment and Use
In general, Im going to be using this [article](https://medium.com/p/30f2e87def1b#523f) to guide how I use Google's services.

As mentioned above, my plan is to make this publically availible on the internet. I have decided to use Google's App Engine to host my model once it is built to serve any web traffic. The model will be pushed up with a flask app acting as a way to interact with. It also may be necessary to attach a Google cloud database to this app to store generated articles, but it is not a part of my MVP.

Something that I may add, given I find it necessary for time saving purposes, is a simple github actions CI/CD pipeline to push changes to GCP.


#### Cloud Platform Costs
For the MVP, I'm only planning on using the GCP app engine to host the model and app. As of right now, the database isn't worth considering. Running this app will likely **cost around 0.20** per hour per instance as a conservative estimate. This can be reduced through possible free trial and student allowances and the fact that it won't be accessed by many people most of the time.

Here is a [link](https://cloud.google.com/appengine/pricing) to where I found this pricing info.

Hopefully, the model will be pre-trained by the time I push it so I won't need to worry about model training costs on GCP.


# Risks 
Current risks invloved with this plan are cost and performace related. Transformers are big and expensive so I have to be smart about how I use my resources. 

Currently, my plan is to use the school provided compute resources to do the training I need to do. Given that I can import the word embeddings from glove, I will have a baseline level of performance immediatly. Using a transfer learning approach, I will train my model with subsets of my data to ensure it isn't running for days on school computer. I will track performance gains and adjust what data and how much I'm feeding it over time. 

Another consideration is my GCP budget, as of the start of the project, I have 25 dollars which will definitley not cover multiple days worth of training. All training, if possible, will be done on school machines. I will also have to consider that I want to have my website and possibly a database running on google services so that will also factor into how I use those services. 

Finally, there is also the possiblity that my model my not make great results which would be unfortunate but it would affect the overall build process of the app.