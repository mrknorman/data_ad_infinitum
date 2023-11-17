= The Problem with Parameters <hyperparameters-section>
\
An applicable machine-learning approach can be found for almost every problem in gravitational-wave data science. That does not mean that machine learning should be applied to every problem in gravitational-wave data science. We must be careful with a liberal application of machine learning approaches and always keep the goal in mind: what exactly are we trying to achieve by applying this particular method? As described in the "No free lunch theorem", [https://ieeexplore.ieee.org/document/585893] for every possible algorithm, there are advantages and disadvantages and no algorithm that completely supersedes another in all classes. This means a rigorous and systematic method for comparing different techniques is required. This problem is confounded with deep learning techniques, as the number of free parameters when designing and optimising your neural network is vast - technically infinite in the non-real case that network size is not a constraint. 

There are a huge number of regularisations that can be applied to a network, and the number of developed layer types and general architectures is considerable and increasing in an almost logarithmic fashion year on year. Even ignoring the number of different types of regularisation, most regularisations have multiple associated parameters, including parameters specifying the design of individual network layers. We label any parameter to do with the model design (aka its regularisation), or that is not optimised during the model training process as a "hyperparameter". Hyperparameters include values to do with the greater structure of the network, such as the type and number of layers in a network, the configuration of the layers themselves, the number of neurons in a dense layer, or the number of filters in the convolutional layer; the training of the network, the learning rate, the number of epochs, and the optimiser, as well as all the parameters associated with the training dataset. Essentially hyperparameters encompass all parameters that must be determined before the initialisation of model training.

This chapter will first give a brief overview of available hyperparameter optimisation methods, then discuss why evolutionary population-based methods were chosen as the hyperparameter optimisation technique for the remainder of this thesis, followed by a demonstration of the effectiveness of hyperparameter optimisation to improve model performance on various gravitational-wave data analysis problems. We will conclude by discussing how this work has been pivotal in developing MLy, a machine learning pipeline currently in review for live deployment in the latter half of the fourth joint observing run.

The goal of any given hyperparameter optimisation process is to maximise the model's performance given a specific objective function. This objective function could be as simple as minimising the model loss, but other performance metrics might also be important to us, such as model inference time or memory usage - or, as is the case for gravitational wave transient detection, minimising values that it would not necessarily make sense to have as part of the loss function, like false alarm rate. If we naively gave models a loss function which only allows a once-in-one-hundred-years false alarm rate, they might never produce a positive result at all. It would be hard to balance such a low false alarm rate requirement with other terms in the loss function, and balancing loss function terms is always a difficult challenge that leads to unstable training. [CITATION AND REPHRASING ARE PROBABLY NEEDED HERE.]

If one is to compare two regularisation techniques, for example, comparing fully connected networks to networks with some convolutional layers, a method must be used to determine all of these hyperparameters. Many, if not most, of these hyperparameters, will have some effect, somewhere between significant and small, on the model's overall performance. Thus the vector space defined by these parameters comprises regions of different model performances, and indeed model performance can be measured in multiple ways. Presumably, given the task at hand, there will be some region within this parameter space that maximises desired performance goals. In the optimal scenario, the comparison of two sets of architectures and hyperparameters will occur between these regions. Thus, a method must find approximate values for these optimal hyperparameters. The reader might now see the recursion that has started here. We are applying an optimisation method to an optimisation that will introduce its own set of hyperparameters. Such hyperparameters will, in turn, need to be at least selected if not optimised. However, it can be shown that the selection of network hyperparameters can make a profound impact [cite] on the performance of the model, and it is hoped that with each optimisation layer, the effects are considerably diminished, meaning that roughly tuned hyperparameters for the hyperparameter optimiser are sufficient for to find comparably optimised solutions.

We can use a similar example parameter space that we generated for @gradient-descent-sec, except this time it is being used to represent hyperparameter space against the model-objective function, rather than parameter space against the loss.

Fig: An example parameter space, here represented in  2D. In actuality, the space is likely to have a much larger dimensionality.

Perhaps unsurprisingly, hyperparameter optimisation is an area of considerable investigation and research in machine learning. However, similar to the rest of the field, I think it would be incorrect to call it well-understood. There are several effective methods for hyperparameter optimisation, and there is no universally accepted set of criteria for which method to use for which problems. What follows is a brief and undoubtedly non-comprehensive review of currently available hyperparameter optimisation techniques.

=== Human-guided trial and error

The most straightforward and obvious method to find effective model hyperparameters relies on human-guided trial and error. This method, as might be expected, involves a human using their prior assumptions about the nature of the problem, the dataset, and the model structure, to roughly guide them towards an acceptable solution, using multiple trials to rule out ineffective combinations and compare the results to the human's hypothesised intuitions. Evidently, whilst this technique is simple to implement and can be time efficient, it suffers from several deficiencies. The results of this method can vary in effectiveness depending on the previous experience of the guiding human; if they have a lot of experience with prior optimisation tasks, they are likely to have more effectively tuned priors. It is also possible that an experienced optimiser might have overly tuned priors, and that bias might cause them to miss possible new solutions that were either previously overlooked or are only relevant to the particular problem being analysed. The results of this method also suffer from a lack of consistency; even the most experienced human optimiser is unlikely to perform precisely the same optimisation technique across multiple problems. Despite these weaknesses, this method is commonly used throughout gravitational wave machine-learning papers and can still be an effective solution for isolated optimisation.

=== Grid Search
A more methodical approach is to perform a grid search across the entirety or a specified subsection of the available parameter space. In this method, a grid of evenly spaced points is distributed across the selected parameter space - a trial is performed at each grid point, and the performance results of those trials are then evaluated. Depending on the computing power and time available, this process can be recursed between high-performing points. This method has the advantage of performing a much more rigorous search over the entirety of the parameter space. However, it can be highly computationally expensive if your parameter space has large dimensionality, which is often the case. A grid search can also be ineffective at finding an optimal solution if the objective function is non-linear and highly variable with minor changes or evidently if its solution lies outside of the range of initial boundaries.

Fig: An example of the samples a grid search might use to find an optimal hyperparameter solution.

=== Random Search
Random search is a very similar method to a grid search; however, instead of selecting grid points evenly spaced across the parameter space, it randomly selects points from the entirety of the parameter space. It has similar advantages and disadvantages to grid search, and with infinite computing resources, both would converge on the ground truth value for the objective function. However, random search has some benefits over grid search that allow it to more efficiently search the parameter space with fewer evaluations. When performing a grid search, the separation of grid points is a user-defined parameter, which both introduces a free parameter and creates possible dimensional bias. A grid search will also search the same value for any given hyperparameter many times, as along the grid axis, it will appear many times, whereas a random search should rarely repeat samples on any hyperparameter. It should also be noted that some statistical uncertainty will be introduced, which would not be present in the case of a grid search and might limit the comparability of different approaches. Both the random and grid search techniques have the disadvantage that all samples are independently drawn, and unless the processes are recursed, no information from the performance results can influence the selection of new points.

Fig: An example of the samples a random search might use to find an optimal hyperparameter solution.

=== Bayesian Optimisation
A Bayesian optimisation approach makes use of our initial beliefs, priors, about the structure of the objective function. For example, you might expect the objective function to be continuous and that closer points in the parameter space might have similar performance. 
The objective function is estimated probabilistically across the parameter space. It is updated as more information is gathered by new samples, which can be gathered either in batches or one at a time. The information obtained by these new samples is incorporated into the estimated objective function in an effort to move it closer to the ground truth objective function. 

The placement of samples is determined by a combination of the updated belief and a defined acquisition function, which determines the trade-off between exploration and exploitation. The acquisition function assigns each point in the parameter space a score based on its expected contribution to the optimisation goal, effectively directing the search process. A standard method for modelling the objective function in Bayesian optimisation is Gaussian Processes, but other techniques are available, such as Random Forests and Bayesian Neural Networks, among others. This optimisation technique is often employed when evaluating the objective function is expensive or time-consuming, as it aims to find the optimal solution with as few evaluations as possible.  

Fig: An example of the samples a Bayesian optimisation might use to find an optimal hyperparameter solution.

=== Gradient-Based Optimisation

=== Population-Based methods

Population 

=== Why genetic algorithms?


== Layer Configuration Tests
Testing which combinations of layers are most effective.

=== Dense Layers

=== Convolutional Layers <cnn_sec>

=== Regularisation

=== Custom Layer Exploration

== Input Configuration Tests <input-configuration-sec>
Testing which input method is most effective. I.e. number of detectors and widthwise, lengthwise, or depthwise.

One detector vs multiple.

SNR cutoff point.


=== Noise Type Tests <noise-type-test-sec>
Also, noise type.

And feature engineering.

== Output Configuration Tests

Baysian tests

== Label Configuration Tests
Testing which configuration of the label is the most effective combination of noise, glitch, etc.

== Branched Exploration

== All together

== Deployment in MLy <deployment-in-mly>