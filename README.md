ST558 Project 2: Creating predictive models and automating Markdown
reports.
================
Josh Baber & Lan Lin
2022-07-06

This report will be analyzing and fitting models on the [Online News
Popularity Data
Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
from UC Irvine’s machine learning repository. This data looks at nearly
60 variables and tries to predict the number of shares that an article
will get. We will be performing some basic exploratory data analysis
with tables and graphs, then will fit some models on the data to try to
predict the number of shares an article gets. We subset the number of
predictors to about 30. Many of the predictors in the data set are
indicator variables, which indicate things like day of the week or data
channel. Other important predictors include article word count, number
of videos or images or links in the article, the rate of positive or
negative words, and more. When fitting the models, we split the data
70/30 into training and testing sets, respectively. We will be fitting
four models on the training data: two linear regression models, a random
forest model, and a boosted tree model. At the end, we will be comparing
the root mean square error (RMSE) of each model on the testing set and
decide which one performed the best.

[Lifestyle articles is available
here](https://github.com/oaktreetrail/ST558_Project2/blob/main/Lifestyle.md)
[Lifestyle articles is available
here](https://github.com/oaktreetrail/ST558_Project2/blob/main/Entertainment.md)
[Lifestyle articles is available
here](https://github.com/oaktreetrail/ST558_Project2/blob/main/Business.md)
