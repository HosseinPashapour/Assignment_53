# Assignment 53
# Deep Learning



# 5 Animals Classification 🐼

## Description

There are two python file in animals folder. The file which name is `AnimalClassification.ipynb` is made for creating train and validation data and making model.
- Classification between 4 animals:
   Elephant 🐘
    Dog 🐶
    Cat 🐈 
    Giraffe 🦒
    Pandas 🐼
- Collect more than 200 photo data from each animal for Model training
- Using Augmentation To increase Train data

## Result


### Train
<img src="Animal_Classification\Output\Train.png">


## Training Information

 
 |           |       Loss     |        accuracy     |
 |---------: | :----------------: |:----------------: |
 |    Train            |       0.2864           |        0.8962          |
 |    Validation            |        0.8383         |        0.7510          |


  
### Model Confusion Matrix
<img src="Animal_Classification\Output\Matrix.png">



## How to Install
Run following commend :
```
pip install -r requirments.txt
```


## How to Run
```
Run AnimalClassification.ipynb file in jupyter notebook
```



# 17 Flowers Classification🌹


## Description

+ In this face classification project ,  we have used CNN to predict Flower images labels , and a dataset which contains 17 classes of flowers.



## Result


### Train
<img src="Flower_Classification\Output\Train.png">


## Training Information

- | Dataset |  Loss	| Accuracy 
    | :---         |     :---:      |          :---: |
    |Train  | 0.0442  | 0.9883  |
    |Val     | 2.8415   | 0.5692   |
    |test     |   1.7977 |  0.6265  |




### Model Confusion Matrix
<img src="Flower_Classification\Output\matrix.png">


 - <a href='https://https://t.me/Diagnosis_Flower_bot'>bot telegram</a>
   I use a free host to upload and run my code on it so not to need run this file on your system to use telegram bot.
   This telegram bot has some commands that I wrote them befor and now I add /image command. After send this command, you should send a flower picture and the bot is predicting the name your flower.

The file which name is `Flowersbot.py` is made for creating a telegram bot. You can access this telegram bot with this name:
   @NEW_pylearn_bot


   
## How to Install
Run following commend :
```
pip install -r requirments.txt
```


## How to Run
```
Run Flower_Classification.ipynb file in jupyter notebook
```











