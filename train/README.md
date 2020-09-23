# Training characters

### Use train.py for training
* 1. Modify the path passed by load_chars to your path. The char_good folder here is the training sample I marked.
* 2. The last letter of the file name is used as a label, such as ch1_2_3.jpg, that is, the picture is 3 characters
* 3. After training, digits_svm.dat is the output model

### Note
* 1. The training will use a total sample of 0.9 for training, then use 0.1 for testing, and output the test results
* 2. train2.py has not been completed yet, it will be better if you have time
* 3. The model saved in train.py is svm, and knn also tests it and gives the test results