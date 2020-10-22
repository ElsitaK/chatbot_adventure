# chatbot_adventure
building a chatbot that uses a neural network

Master branch is simplest model and data file. 
Model branch has chatbot trained from movie quotes dataset (from the Cornell Movie Dialogs Corpus)
Upcoming: New branch where the movie lines will be clustered using Kmeans prior to training.

COMPLETED: 

- make new "model" branch for gui and modify
- either modify intents.json directly or find new data to train model
- learn about JSON file formatting
- convert movie lines csv data to json format
- train new neural network model on movies data
- run chatbot using the new model

TO DO:

-> on new branch
- cluster 10000 movie lines via kmeans (decide on number of groups, which will translate to number of possible responses)
- rewrite csv and then json based on clusters
- train new neural network model with clusters as labels
- run chatbot using the new model
