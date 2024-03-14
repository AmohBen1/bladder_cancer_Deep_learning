# Pretrained models.
## Downloading the Pretrained Model

The ImageNet pre-trained models are often good choices for computer vision transfer learning, 
as they have learned to classify various different types of images. In doing this, they have 
learned to detect many different types of features that could be valuable in image recognition. 
Because ImageNet models have learned to detect animals, including dogs, it is especially well 
suited for this transfer learning task of detecting Bo.

Let us start by downloading the pre-trained model. Again, this is available directly from the Keras library. 
As we are downloading, there is going to be an important difference. The last layer of an ImageNet model is a 
dense layer of 1000 units, representing the 1000 possible classes in the dataset. In our case, we want it 
to make a different classification: is this Bo or not? Because we want the classification to be different, 
we are going to remove the last layer of the model. We can do this by setting the flag include_top=False when 
downloading the model. After removing this top layer, we can add new layers that will yield the type of 
classification that we want:

## Freezing the Base Model
Before we add our new layers onto the pre-trained model, we should take an important step: freezing the model's 
pre-trained layers. This means that when we train, we will not update the base layers from the pre-trained model. 
Instead we will only update the new layers that we add on the end for our new classification. We freeze the initial 
layers because we want to retain the learning achieved from training on the ImageNet dataset. If they were unfrozen 
at this stage, we would likely destroy this valuable information. There will be an option to unfreeze and train 
these layers later, in a process called fine-tuning.

Freezing the base layers is as simple as setting trainable on the model to False.

## Adding New Layers
We can now add the new trainable layers to the pre-trained model. They will take the features from the 
pre-trained layers and turn them into predictions on the new dataset. We will add two layers to the model. 
First will be a pooling layer like we saw in our earlier convolutional neural network)). We then need to add 
our final layer, which will classify Bo or not Bo. This will be a densely connected layer with one output.

## Compiling the Model
We need to compile the model with loss and metrics options. We have to make some different choices here. 
In previous cases we had many categories in our classification problem. As a result, we picked categorical 
crossentropy for the calculation of our loss. In this case we only have a binary classification problem (Bo or not Bo), and so we will use binary crossentropy. Further detail about the differences between the two can found here. 
We will also use binary accuracy instead of traditional accuracy.

By setting from_logits=True we inform the loss function that the output values are not normalized (e.g. with softmax).

Important to use binary crossentropy and binary accuracy as we now have a binary classification problem

## Augmenting the Data
Now that we are dealing with a very small dataset, it is especially important that we augment our data. 
As before, we will make small modifications to the existing images, which will allow the model to see a 
wider variety of images to learn from. This will help it learn to recognize new pictures of Bo instead 
of just memorizing the pictures it trains on.