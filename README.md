ST558 Project 2: Creating predictive models and automating Markdown
reports.
================
Josh Baber & Lan Lin
2022-07-06

## Brief Description of Repo 

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

To automate the creation of the documents, we created a data frame of channels and channel file names.  We had to specify the channel file names in the YAML header like so:

`params:  
  channel:  
    label: "Data Channel"  
    value: Lifestyle  
    input: select  
    choices: [Lifestyle, Entertainment, Business, Social Media, Tech, World]`
    
Then, after creating a data frame with two columns, one for file name and one for channel name, we could submit this code to render all six documents:

`library(rmarkdown)  
apply(reports, MARGIN = 1,  
      FUN = function(x){  
        render(input = "ST558_Project2.Rmd",  
               output_file = x[[1]],  
               params = x[[2]],  
               output_format="github_document",  
               output_options=list(html_preview=FALSE, toc=TRUE, toc_depth=2))  
      })`

## Packages Used 

We used the following packages to complete the project:

-   [tidyverse](https://www.tidyverse.org/) for data cleaning and transforming with dplyr, and plotting with ggplot2.
-   [reader](https://cran.r-project.org/web/packages/reader/reader.pdf) for reading files.
-   [corrplot](https://www.rdocumentation.org/packages/corrplot/versions/0.92) for creating a detailed correlation plot of variables.
-   [caret](https://topepo.github.io/caret/) for training our models, we used caret to train all four of our models.
-   [elasticnet](https://cran.r-project.org/web/packages/elasticnet/elasticnet.pdf) a package that is necessary for the boosted tree model.
-   [ggridges](https://cran.r-project.org/web/packages/ggridges/ggridges.pdf) for creating a plot that has multiple density plots in "ridges".
-   [gridExtra](https://cran.r-project.org/web/packages/gridExtra/gridExtra.pdf) for structuring some tables.
-   [doParallel](https://cran.r-project.org/web/packages/doParallel/doParallel.pdf) for parallel processing, particularly when we get to the random forest model and boosted model since they are computationally expensive.
-   [microbenchmark](https://cran.r-project.org/web/packages/microbenchmark/microbenchmark.pdf) to see how long different tasks take.

## Links to Each Article.

Below are the links to each of the articles we produced.

-   [Lifestyle article is available here](Lifestyle.html)
-   [Entertainment article is available here](Entertainment.html)
-   [Business article is available here](Business.html)
-   [Social Media article is available here](Social%20Media.html)
-   [Tech article is available here](Tech.html)
-   [World article is available here](World.html)
