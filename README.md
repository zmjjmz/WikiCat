# WikiCat
Code to train and evaluate a Wikipedia page categorizer


## Code Structure

Run `WikiCatBuild.py category_uri_file [options]` to scrape a list of Categories

Options:
        category_uri_file File containig a newline separated list of URIs to Cateogry pages
		-h Print this
		-v [0,1,2,3] Set verbosity level. Defaults to 1.
        -r Root directory where model, representer, cache, and GloVe vectors will be stored.
           Make sure there's at least 3GB available for the GloVe vectors.

Run `WikiCatClassify.py uri [options]` afterwards with the arguments specified to obtain a list of category probabilities for the page specified
Options:
        uri 
		-h Print this
		-v [0,1,2,3] Set verbosity level
		-r Root directory containing representer and model.

IPython Notebooks. Run WikiCatBuild with -r test at least once to use these notebooks unmodified
- Scraper No Scraping! : This is the notebook I used to prototype the scraping process
- Exploratory : This is the notebook I used to to prototype the process of finding a decent classifier for the data
- Analysis : This is a notebook going through some properties of the data and (to a lesser extent) the learned classifier

##Installation

Note: So far I can only support Python 3.

    git clone https://github.com/zmjjmz/WikiCat.git
    cd WikiCat/
    pip install -r requirements.txt

The scripts and notebook should be useable as outlined above.
