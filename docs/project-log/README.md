# Project Log 


## 2/10
Looking into transformers rather than using gans. I found a pre-trained embedding space for which will allow me to not have to train one from scratch. [Link](https://github.com/stanfordnlp/GloVe) to it here. Another [blog](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec) about how to get started with transformers in pytorch.

I'm also looking into google vertex AI as a possible service to host my built model for when I want to push it. 
[Link](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) to google clouds Vertex API documentation. Another [Link](https://medium.com/@piyushpandey282/model-serving-at-scale-with-vertex-ai-custom-container-deployment-with-pre-and-post-processing-12ac62f4ce76) to a blog post about how to get a custom container pushed to their services

## 2/14
Rather than vertex AI, I'm now looking at Google's app engine to run my flask app. This [article](https://medium.com/gft-engineering/everything-you-wanted-to-know-about-serving-language-models-on-gcp-but-were-afraid-to-ask-30f2e87def1b#2468) has given me a bit of insight into how I should probably be setting up my model on googles platforms given the nature of my transformer project.

I have also started working on the initial pytorch tranformer, I am downloading the stanford GloVe dataset to import their word embeddings and see how they perform at a basic level.

I have also finished my project proposal, a more in depth description of what the project will be can be found there.


## 2/22 
As of right now I have been digging into how transformers work and how to set one up for the needs of my project. 

Important links to articles I've been using 
-  [Basic tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) for pytorch transformer classification
-  [Harvard Paper](https://nlp.seas.harvard.edu/2018/04/03/attention.html#data-loading) with annotated details on how to build a transformer from scratch and how each part works
-  [Blog Post](https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1) on how to get started with pytorch transformers as a beginner 


I've also been working on making it possible for me to train my model remotely through the Gonzaga provided server. This has included setting up a way to save my trained model state with pytorch using their [built in functionality](https://pytorch.org/tutorials/beginner/saving_loading_models.html). Also I've been looking into **screen** as a way to ssh into their network to use their compute resources, [here is a link](https://linuxize.com/post/how-to-use-linux-screen/) to use it at a basic level.

# 3/25 

Interesting video talking about Stanford's Alpaca. This language model, as they describe, behaves similarly to Open-AI's chaptGPT while being small and easy to reproduce.

[Link](https://www.youtube.com/watch?v=xslW5sQOkC8&list=LL&index=1) to video.

It might be worth looking into some way to set up my project similar to thiers in order to get better performance out of my model. With their setup I can use chatGPT to train my model rather than create my own RL model.

# 4/25
As I work through this project, its goals are beginning to shift towards a more exploratory project. To use chatgpt for posts I would need to find a subreddit which works primarily off of text. I have settled on r/AITA due to its structured text data. But because a lot of subreddit's don't allow bots to post(if they find out) I have to be careful how much I'm allowing mine to post to collect data. This has slowed down my analysis of performance a bit. 