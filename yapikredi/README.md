For this case, there are given signatures from different people with labels. So this was the classification task. 

I have first added methods to normalize and augment the dataset to increase the correctness of model. I have used also PCA to reduce dimensionality of image dataset. There are two models I have tried. 

I assume these methods to augment image dataset is already built in some other library, maybe in Keras or Tensorflow. Since I don't know about them I tried to built them manually.

I have used 60 * 60 pixel dimension for my calculations. I have experienced first time the computation power requirement for such tasks.

The first one is Support Vector Machines classifier with RBF kernel. It simulates infinite polynomial degree feature space which gives score of 0.85. There are other kernels also could be used. I haven't tried them.

Than I used Neural Network Classifier with one layer (basic neural network) with invscaling learning rate. I got the result of 0.79 if I remember correctly.  


