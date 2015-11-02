from __future__ import division, print_function # Python2 compat
from argparse import ArgumentParser
from os import getcwd, mkdir
from os.path import (
    join,
    exists,
    getmtime,
)
from WikiCatUtils import (
    #load_model_params,
    get_fullpath,
    read_categories,
    #save_model,
    ArticlePage,
    CategoryPage,
    Cache,
    Representer,
)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action='store', help='Set\
                             verbosity level to one of [0 - 3], where 0 is\
                             quiet. Defaults to 1.', dest='verbosity', type=int, default=1)

    arg_parser.add_argument("categories_file", action='store', help='Path to\
                             CSV containing Category Page URIs, Max Amount of\
                             Articles (e.g. Philosophy, 100). Non-absolute paths are\
                             considered relative to the current directory.')#, dest='categories')

    """
    arg_parser.add_argument("-r", "--representation", action='store', help='How to\
                            represent the data to the model. Options are BoW,\
                            GloVe, word2vec. Defaults to BoW', dest='repr', default='BoW')

    arg_parser.add_argument("-m", "--model", action='store', help='Path to JSON\
                            file containing a model / hyperparameter\
                            specification as outlined in ModelChoices.txt. Non-absolute paths are\
                            considered relative to the current directory.\
                            Defaults to Logistic Regression w/C=1.0', dest='model', default='./default_model.json')
    """
    arg_parser.add_argument("-o", "--model-out", action='store', help='Path to\
                            where the trained model should be saved. Defaults\
                            to /tmp/WikiCat/model.pkl. Non-absolute paths are\
                            considered relative to the current directory.', dest='model_out',
                            default='/tmp/WikiCat/model.pkl')

    arg_parser.add_argument("-c", "--cache", action='store', help="Directory containing the\
                            scraped articles in subdirectories corresponding to their\
                            categories.  When the article scraping happens, these articles\
                            are used if their date modified is greater than the article's,\
                            and the scraping writes to this directory. If left unspecified,\
                            articles will be stored in /tmp/WikiCat/cache/.",
                            dest='cache', default='/tmp/WikiCat/cache')

    arg_parser.add_argument("-u", "--use-only-cache", action='store_true',
                            help="If set, will only load existing files and not\
                            download any new ones. It will still attempt to\
                            download category pages. If a new category is in the\
                            categories_file it will not download any of the new\
                            categories but it will make the folder for it.",
                            dest='cache_only', default=False)

    arg_parser.add_argument("--maxlinks", action='store',
                            help="Maximum amount of links that get used for\
                            each category. Can be used to help control bias, but\
                            at the cost of having less data. If not set, no\
                            maximum amount of links is imposed",
                            dest='maxlinks', default=None, type=int)



    args = arg_parser.parse_args()

    # Setup data
    if not exists('/tmp/WikiCat'):
        mkdir('/tmp/WikiCat')
    categories_to_download = read_categories(get_fullpath(args.categories_file))
    cache = Cache(get_fullpath(args.cache), verbosity=args.verbosity)
    for category_uri in categories_to_download:
        cache.loadCategory(category_uri, maxlinks=args.maxlinks, only_use_cached=args.cache_only)


    # Build training / validation / testing set.
    # so now we have a Cache object with a content dictionary of articleName ->
    # article Title
    # We need to make a dataset of train / validation / test
    # We'll default to a 60/20/20 split

    # We'll sample each category in the 60/20/20 split to make sure we don't
    # accidentally omit a category since they have very a very different amount
    # of articles associated with them

    dset = Cache.get_dataset(0.6, 0.2, 0.2)
    # each of train, val, test is a tuple of article content and labels

    # Build representation of the data according to the arguments
    # this is encapsulated in the Representer object, which will take care of
    # vocabulary building and term weighting (if necessary)
    dataset_rep = Representer() # getting rid of options for this
    #sklearn style
    dset_represented = {}
    dset_represented['train'] = dataset_rep.fit_transform(dset['train'])

    # Train model

    dset_represented['val'] = dataset_rep.transform(dset['val'])
    dset_represented['test'] = dataset_rep.transform(dset['test'])

    model = Model()
    # val is used for hyperparameter tuning
    model.fit(dset_represented['train'], dset_represented['val'])

    # Evaluate model on validation, testing sets
    if args.verbosity > 0:
        print(model.evaluate(dset_represented))

    model.save(args.model_out)


