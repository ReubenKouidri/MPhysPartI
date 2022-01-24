# ecg_classification
...still in the making...

Contains the code used to process the data and train the model, as well as generating answers to the 'test-set' and scoring.

In its current state, for the code to work the dataset will need to be imported from the CPSC-2018 website: http://2018.icbeb.org/Challenge.html
I recreated 'TrainingSet1' to contain the first 100 samples to use whilst testing whether the code was working correctly.
The 'full dataset' is the union of all training sets

The code:
  1. 'ArrhythmiaDataset2D.py' contains the Dataset class which imports and processes the ECG signals, returning images and targets. To be used in conjunction with a       Dataloader object.
  2. 'model.py' contains the model architecture
  3. 'train.py' uses these 2 files to train and save the model
  4. 'test.py' imports the trained model and tests it on 300 validation files, outputting a .csv file called 'answers.csv'
  5. 'score.py' will then score these answers, returning a .csv file containing the scores to the cardiac arrhythmia super-types.
  *'reference.csv'* contains the true labels for the corresponding files
  
p.s. As it stands, the code will suffer refactoring problems. This is resolved in Colab, but I have not yet transferred this over due to time.
