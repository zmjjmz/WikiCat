# WikiCat
Code to train and evaluate a Wikipedia page categorizer


## Code Structure

Run `WikiCatBuild.py [options]` to scrape a list of Categories

Options:
		-h Print this
		-v [0,1,2,3] Set verbosity level. Defaults to 1.
		-c a CSV of Category Page URIs, Max Amount of Articles (e.g. Philosophy, 100)
		-e The model, one of {tbd, tbd, or tbd}, if left unspecified it will default to "bow"
		-m P2 Model hyperparameters in JSON. Options and their defaults are specified in Models.md
		--cache P2 Directory containing the scraped articles in subdirectories corresponding to 
						their categories.  When the article scraping happens, these articles are used
						if their date modified is greater than the article's, and the scraping writes
						to this directory. If left unspecified, articles will be stored in /tmp.

		-o Output path for the model to be serialized. If left unspecified, it will be stored in /tmp.

Run `WikiCatClassify.py [options] uri` afterwards with the arguments specified to obtain a list of category probabilities for the page specified
Options:
		-h Print this
		-v [0,1,2,3] Set verbosity level
		-e Model type (same options as -e for WikiCatBuild)
		-m Path to serialized model

The IPython notebooks included are artifacts of the prototyping process.

