# Project Log 


## Week of 2/6

### 2/10
Looking into transformers rather than using gans. I found a pre-trained embedding space for which will allow me to not have to train one from scratch. [Link](https://github.com/stanfordnlp/GloVe) to it here. Another [blog](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec) about how to get started with transformers in pytorch.

I'm also looking into google vertex AI as a possible service to host my built model for when I want to push it. 
[Link](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) to google clouds Vertex API documentation. Another [Link](https://medium.com/@piyushpandey282/model-serving-at-scale-with-vertex-ai-custom-container-deployment-with-pre-and-post-processing-12ac62f4ce76) to a blog post about how to get a custom container pushed to their services

### 2/14
Rather than vertex AI, I'm now looking at Google's app engine to run my flask app. This [article](https://medium.com/gft-engineering/everything-you-wanted-to-know-about-serving-language-models-on-gcp-but-were-afraid-to-ask-30f2e87def1b#2468) has given me a bit of insight into how I should probably be setting up my model on googles platforms given the nature of my transformer project.

I have also started working on the initial pytorch tranformer, I am downloading the stanford GloVe dataset to import their word embeddings and see how they perform at a basic level.

I have also finished my project proposal, a more in depth description of what the project will be can be found there.
