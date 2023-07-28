# Project Structure

## Documentation Structure 
All of the documentation created for the project will be stored within the docs folder. The **project proposal**, **project log** and **research proposal** will have their own folders with a readme. When in Github you can navigate to these folders and read them from there. 

Link to [project proposal](https://github.com/cstainsby/B-Onion/tree/main/docs/project-proposal)

Link to [project log](https://github.com/cstainsby/B-Onion/tree/main/docs/project-log)

Link to [research proposal](https://github.com/cstainsby/B-Onion/tree/main/docs/research-proposal)

## Post Development Reflections
During development I encoutered many roadblocks which I had to accomodate for which radically changed how I was doing my project. Unfortunatley the following happened:
1. The twitter API became prohibitivley difficult to get access to so I switched to reddit's api.
1. In order to achieve meaningful results, I began using chatGPT's pretrained model api. They removed free credits though which made it prohibitivly expensive for the scale of my project to use. This forced me to scale back the scope of my project to only focus on replying to AmITheAsshole posts using huggingface BERT models. 
1. Changes to the reddit API made retriving large amounts of data prohibitivley expensive as well making it far more difficult to train a data hungry model like the transformer I was using. I had to again settle for just classification of the AITA posts. 

## Performance of the Classifier
My classifier did actually perform somewhat well achieving an on average **80% accuracy**. However this is very likely due to the overrepresentation of one type of classification (NTA). The data I pulled was representative of this trend which made it difficult for the classifier to recognize when to guess anything other than NTA.

Unfortuanetly due to the data limits set in place by reddit, I will likely not be able to get a sizeable amount of data which represents other classifications to train on.

## Future Goals
I would still like to add the functionality to generate responses to posts. Using a pretrained model from HuggingFace would be helpful but the vocabulary used by the subreddit is very unique and would make it difficult to pass my responses off as human.

I would also like to make a perfomance dashboard where I could A/B test performance of different model versions to help

I would also like more data. Unfortunatley with the changes this has become difficult to do. My current data is very overrepresentative of a specific classification and it does a poor job of recognizing sentiment. 

