# Selecting and Training a Model

## Modeling overview

Key question: What are they key challenges in building a deployment
ready model?

-   Select and train model
-   Perform error analysis

Two approaches to the problem

-   Model-centric AI development
-   Data-centric AI development

Practical projects will benefit more from choosing data-centric: more
quality data is more important.

## Key Challenges

Framework: AI system = Code + Data

-   Code: algorithm/model
    -   Model-centric, same data but change the model
    -   Hyper-parameter is also an choice variable, but the search space
        is relatively narrow
-   Data
    -   Data-centric, practical problems usually only require off the
        shelf models but require customised data

Key to improve performance: efficient iterative process to update model,
training and perform error analysis

Bonus: perform final audit on error analysis

Challenges in model development

1.  Doing well on training set
    -   If model is not performing on training set, then most likely it
        will not perform on test set
2.  Doing well on dev/test sets
    -   Research generally focus on this. However, it is insufficient
        for business team
3.  Doing well on business metrics/ project goals

## Why low average error isn’t good enough

Performing well on test set isn’t enough

Challenges

-   Data/model drift
-   Performance on disproportionately important examples
    -   E.g. web search: informational and transactional queries can
        have lower performance. Navigational queries (very specific
        search) must have very accurate results else risking trust.
    -   Changing/imposing the weight of different examples might work
        for some problems, but not all
-   Performance on key slices of the dataset
    -   E.g. ML for loan approval should not discriminate based on
        attributes. High average accuracy is insufficient if there is
        bias towards a racial group.
    -   E.g. Product recommendations from retailers. High average but
        ignores racial group or small retailers or product categories
        will damage the business since you will lose some groups of
        customers.
-   Rare classes
    -   Skewed data distribution. For example, in medical applications
        where positive class is very low.

## Establish a baseline

Baseline is required to know where to improve on.

Set baseline by the type of use case (e.g. Speech recognition)

-   Compare against human level performance, and investigate the
    potential gain in benefit to achieve human level performance
-   Unstructured and structured data
    -   Human perform better in unstructured data (image, audio, text).
        So establishing baseline with human benchmark is good.
    -   Structured data in a giant database confuses human. Human
        baseline is not very useful.

Ways to establish a baseline

-   Human level performance (HLP)
-   Literature search for state-of-the-art/open source
-   Quick-and-dirty implementation
-   Performance of older system

Baseline helps to indicates what might be possible. In some cases is
also gives a sense of what is irreducible error/Bayes error.

For example, it might be near impossible to beat HLP.

## Tips for getting started

Getting started on modeling

-   Literature search to see what’s possible (courses, blogs,
    open-source projects)
    -   If your goal is not research, then pick something reasonable
        that can be start quickly
    -   Do not have to use the latest invention
-   Find open-source implementations if available
-   A reasonable algorithm with good data will often outperform a great
    algorithm with no so good data
    -   Starting early means more time in iterations

Deployment constraints when picking a model

-   Should you take into account deployment constraints when picking a
    model?
    -   Yes if baseline is established and goal is to build and deploy
    -   No if purpose is to establish a baseline and determine what is
        possible and might be worth pursuing

Sanity-check for code and algorithm

-   Try to overfit a small training dataset before training on a large
    one.
    -   For example: predict one single speech, image segmentation to
        make sure if it can at least overfit the training example before
        scaling up. Or image classification just train a subset of 10
        images to make sure at least the basic case is handled

# Error analysis

What is the most efficient use of the time to improve learning algorithm

An example: speech recognition

-   Listen to hundreds of mislabeled examples. Then observe hypothesis
    of errors (e.g. car noise, people noise). New ideas can be proposed
    to reflect the common observations of the errors.
-   This process helps you to understand the category of the source of
    the errors. Generally it’s a manual process.
-   There are emerging automation tools to improve the process of
    debugging.

Iterative process of error analysis

-   Examine/tag examples
    -   Specific class labels: scratch, dent etc
    -   image properties: blurry, dark background, light background etc
    -   other meta-data: phone model, factory
-   Propose tags
    -   User demographic

Useful metrics for each tag

-   What fraction of errors has that tag?
    -   If tags belong to 12% of the error, then maximum possible
        improvement is 12%
-   Of all data with that tag, what fraction is misclassified?
    -   Tells the model accuracy belonging to that tag
-   What fraction of all the data has that tag?
    -   Tells relatively the importance of the tag
-   How much room for improvement is there on data with that tag?
    -   E.g. compare against human

## Prioritizing what to work on

Observe

1.  Gap between model and human level accuracy
2.  % of data

The improvement is (1) \* (2). For example, 1% gap \* 60% data = 0.6%
improvement

Decide on the most important categories to work on based on

-   How much room for improvement there is
-   How frequently that category appears
-   How easy is to improve accuracy in that category
-   How important it is to improve in that category

Adding/improving data for specific categories

-   To improve the model performance on categories you want to
    prioritize
    -   Minimise and focus on the type of data to collect
-   Collect more data
-   Use data augmentation to get more data
-   Improve label accuracy/ data quality

## Skewed datasets

Ratio of positive to negative class is high

Examples

-   Manufacturing: 99.7% no defect (y=0)
    -   Printing 0 will achieve 99.7% accuracy
-   Medical diagnosis: 99% no disease
-   Speech Recognition: wake word detection not spoken 96.7% of the time

More useful to investigate the confusion matrix: precision and recall.

-   Precision: true positive / predicted positive (TP + FP)
    -   Among predictions, percentage of accuracy
-   Recall: true positive / all positive class (TP + FN)
    -   Among all positive, how many are captured
-   F1 score
    -   Combining prediction and recall: doing well on both, F1 = 2 /
        (1/P + 1/R), harmonic mean -&gt; the mean that puts more weight
        on the smaller value
    -   Weights can be changed according to use case
    -   Useful for multi-class metrics as well
        -   Factories might concern more about recall than prediction:
            humans can re-verify the defects
        -   Observe the F1 score for each of the type of defects

## Performance auditing

After the validation of test set F1 score validation.

Auditing framework, check for accuracy, fairness/bias, and other
problems

-   Brainstorm the ways the system might go wrong
    -   Performance on subset of data (e.g. ethnicity, gender)
    -   How common are certain errors (e.g. FP, FN)
    -   Performance on rare classes
-   Establish metrics to assess performance against thesis issues on
    appropriate slices of data
    -   Instead of the full data, analysis performance using a subset
        (slice) of the data
-   Automatic metric computation can be useful to improve the efficiency
    of performance auditing
-   Get business product owner to buy-in on the limitations

An example: speech recognition

-   Brainstorm the ways the system might go wrong
    -   Accuracy on different genders and ethnicities
    -   Accuracy on different devices
    -   Prevalence of rude mis-transcriptions (e.g. GAN to gun, gang).
        Transcription into rude words is more serious than inaccurate
        transcription
-   Establish metrics to assess performance against these issues on
    appropriate slices of data
    -   Mean accuracy for different genders and major accents
    -   Mean accuracy on different devices
    -   Check for prevalence of offensive words in the output

For high-stake application, having a team to brainstorm the possible
errors is often helpful than individual contributors.

# Data Iteration

How to take a data centric AI development

-   Model centric view: take the data and develop model that does as
    well as possible on it (e.g. academic research perform benchmark
    based on a fixed data)
    -   Hold the data fixed and iteratively improve the code/model
-   Data centric view: the quality of the data is paramount. Use tools
    to improve the data quality; this will allow multiple models to do
    well. Hold code fix, change data.
    -   More useful for common use case.

## Data augmentation

Check list

1.  Does it sound realistic
2.  Is the x to y mapping clear (e.g. can humans recognise speech?)
3.  Is the algorithm currently doing poorly on it?

A useful picture of data augmentation: e.g. speech recognition

-   Different types of speech input (e.g. car, plane, train, machine
    noise)
-   Consider a plot with y-axis: performance; x-axis: space of possible
    inputs
    -   The different inputs will have different performance. E.g.
        better performing with car noise than plane noise.
    -   Consider a one-dimensional curve, human and AI will have
        different level of performance (HLP vs Model). The gap
        illustrates an opportunity of improvement. The adjacent problems
        will likely be lessen as well.
    -   The one-dimensional curve is also the error analysis to
        understand where to “pull up” next to achieve near HLP.

Data augmentation is especially useful for unstructured data problems.
The best practices for data augmentation include:

-   Synthetic training example: voice signal + noise
    -   Decision: what kind of background noise, how loud should the
        noise be
-   Goal: create realistic examples that (i) the algorithm does poorly
    on, but (ii) humans (or other baseline) do well on.
    -   If the example is impossible for human to well on, then there is
        no point to train
    -   If the example is already performing well, then there is no
        point to do augmentation as well.
-   Although you can re-train the model with different parameters. It is
    more beneficial to generate augmented data and see if the
    performance improve

Another example: image

-   Contrast changes, Darken changes (but if it is too dark even human
    cannot identify it, then there is no point to implement it as a data
    augmentation).
-   Data augmentation can be done with photoshop. E.g. a scratch using
    photoshop
    -   GAN models can be over-kill when simple photoshop can generate
        the augmented samples

Data iteration loop

-   Add/Improve data (holding model fixed)
-   Training
-   Error analysis

Can adding data hurt?
