## ---- echo = FALSE, message = FALSE, warning = FALSE---------------------
library(tidyverse)
library(haven)
library(knitr)
options(dplyr.print_min = 5)
options(tibble.print_min = 5)
opts_chunk$set(message = FALSE, cache = TRUE)

# Create a channel variable 
channel <- c('Lifestyle', 'Entertainment', 'Business', 'Social Media', 'Tech', 'World')

#create filenames
output_file <- paste0(channel, ".html")

#create a list for each channel with just the team name parameter
params = lapply(channel, FUN = function(x){list(channel = x)})

#put into a data frame 
reports <- tibble(output_file, params)

## ------------------------------------------------------------------------
reports

## ---- eval = FALSE, echo = TRUE------------------------------------------
library(rmarkdown)
## #need to use x[[1]] to get at elements since tibble doesn't simplify
apply(reports, MARGIN = 1,
      FUN = function(x){
        render(input = "Lan.Rmd", output_file = x[[1]], params = x[[2]])
      })
