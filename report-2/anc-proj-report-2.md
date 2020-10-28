# ANC Project Report 2 - Linus

In unsupervised learning, labels are not available. Therefore, the task of the AI agent is not well-defined, and performance cannot be so clearly measured. Consider the email spam filter problem—this time without labels. Now, the AI agent will attempt to understand the underlying structure of emails, separating the database of emails into different groups such that emails within a group are similar to each other but different from emails in other groups.

This unsupervised learning problem is less clearly defined than the supervised learning problem and harder for the AI agent to solve. But, if handled well, the solution is more powerful.

Here’s why: the unsupervised learning AI may find several groups that it later tags as being “spam”—but the AI may also find groups that it later tags as being “important” or categorize as “family,” “professional,” “news,” “shopping,” etc. In other words, because the problem does not have a strictly defined task, the AI agent may find interesting patterns above and beyond what we initially were looking for.

Moreover, this unsupervised system is better than the supervised system at finding new patterns in future data, making the unsupervised solution more nimble on a go-forward basis. This is the power of unsupervised learning.

These groups (known as clusters) should be homogeneous and distinct. In other words, the members within a group should be very similar to each other and very distinct from members of any other group.

Clustering is an unsupervised learning approach, and, therefore, labels are not used. However, we will use the pre-defined labels to judge the goodness of our clustering algorithm at finding distinct and homogeneous groups in this dataset.

The core concept of an autoencoder is similar to the concept of dimensionality reduction we studied in Chapter 3. Similar to dimensionality reduction, an autoencoder does not memorize the original observations and features, which would be what is known as the identity function. If it learned the exact identity function, the autoencoder would not be useful. Rather, autoencoders must approximate the original observations as closely as possible—but not exactly—using a newly learned representation; in other words, the autoencoder learns an approximation of the identity function.

Since the autoencoder is constrained, it is forced to learn the most salient properties of the original data, capturing the underlying structure of the data; this is similar to what happens in dimensionality reduction. The constraint is a very important attribute of autoencoders—the constraint forces the autoencoder to intelligently choose which important information to capture and which irrelevant or less important information to discard.

Autoencoders have been around for decades, and, as you may suspect already, they have been used widely for dimensionality reduction and automatic feature engineering/learning. Today, they are often used to build generative models such as generative adversarial networks.

Undercomplete Autoencoders
In the autoencoder, we care most about the encoder because this component is the one that learns a new representation of the original data. This new representation is the new set of features derived from the original set of features and observations.

We will refer to the encoder function of the autoencoder as h = f(x), which takes in the original observations x and uses the newly learned representation captured in function f to output h. The decoder function that reconstructs the original observations using the output of the encoder function is r = g(h).

As you can see, the decoder function feeds in the encoder’s output h and reconstructs the observations, known as r, using its reconstruction function g. If done correctly, g(f(x)) will not be exactly equal to x everywhere but will be close enough.

How do we restrict the encoder function to approximate x so that it is forced to learn only the most salient properties of x without copying it exactly?

We can constrain the encoder function’s output, h, to have fewer dimensions than x. This is known as an undercomplete autoencoder since the encoder’s dimensions are fewer than the original input dimensions. This is again similar to what happens in dimensionality reduction, where we take in the original input dimensions and reduce them to a much smaller set.

Constrained in this manner, the autoencoder attempts to minimize a loss function we define such that the reconstruction error—after the decoder reconstructs the observations approximately using the encoder’s output—is as small as possible. It is important to realize that the hidden layers are where the dimensions are constrained. In other words, the output of the encoder has fewer dimensions than the original input. But the output of the decoder is the reconstructed original data and, therefore, has the same number of dimensions as the original input.

When the decoder is linear and the loss function is the mean squared error, an undercomplete autoencoder learns the same sort of new representation as PCA, a form of dimensionality reduction we introduced in Chapter 3. However, if the encoder and decoder functions are nonlinear, the autoencoder can learn much more complex nonlinear representations. This is what we care about most. But be warned—if the autoencoder is given too much capacity and latitude to model complex, nonlinear representations, it will simply memorize/copy the original observations instead of extracting the most salient information from them. Therefore, we must restrict the autoencoder meaningfully enough to prevent this from happening.

Variational Autoencoder
So far, we have discussed the use of autoencoders to learn new representations of the original input data (via the encoder) to minimize the reconstruction error between the newly reconstructed data (via the decoder) and the original input data.

In these examples, the encoder is of a fixed size, n, where n is typically smaller than the number of original dimensions—in other words, we train an undercomplete autoencoder. Or n may be larger than the number of original dimensions—an overcomplete autoencoder—but constrained using a regularization penalty, a sparsity penalty, etc. But in all these cases, the encoder outputs a single vector of a fixed size n.

An alternative autoencoder known as the variational autoencoder has an encoder that outputs two vectors instead of one: a vector of means, mu, and a vector of standard deviations, sigma. These two vectors form random variables such that the ith element of mu and sigma corresponds to the mean and standard deviation of the ith random variable. By forming this stochastic output via its encoder, the variational autoencoder is able to sample across a continuous space based on what it has learned from the input data.

The variational autoencoder is not confined to just the examples it has trained on but can generalize and output new examples even if it may have never seen precisely similar ones before. This is incredibly powerful because now the variational autoencoders can generate new synthetic data that appears to belong in the distribution the variational autoencoder has learned from the original input data. Advances like this have led to an entirely new and trending field in unsupervised learning known as generative modeling, which includes generative adversarial networks. With these models, it is possible to generate synthetic images, speech, music, art, etc., opening up a world of possibilities for AI-generated data.
, we will use the labels.

Unsupervised Learning
Unsupervised learning has not had nearly as many successes to date as supervised learning has had, but its potential is immense. Most of the world’s data is unlabeled. To apply machine learning at scale to tasks that are more ambitious in scope than the ones supervised learning has already solved, we will need to work with both labeled and unlabeled data.

Unsupervised learning is very good at finding hidden patterns by learning the underlying structure in unlabeled data. Once hidden patterns are uncovered, unsupervised learning can group the hidden patterns based on similarity such that similar patterns are grouped together.

Once the patterns are grouped this way, humans can sample a few patterns per group and provide meaningful labels. If the groups are well-defined (i.e., the members are homogeneous and distinctly different from members in other groups), the few labels that humans provide by hand can be applied to the other (yet unlabeled) members of the group. This process leads to very fast and efficient labeling of previously unlabeled data.

In other words, unsupervised learning enables the successful application of supervised learning methods. This synergy between unsupervised learning and supervised learning—also known as semisupervised learning—may fuel the next wave in successful machine learning applications.

Clustering makes labeling unlabeled data considerably more efficient. Because similar data is grouped together, a human needs to label only a few of the points per cluster. Once a few points within each cluster are labeled, the other not-yet-labeled points could adopt the labels from the labeled points.
