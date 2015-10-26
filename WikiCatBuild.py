from __future__ import division, print_function # Python2 compat
from argparse import ArgumentParser
from os import getcwd
from os.path import (
    join,
    exists,
    getmtime,
)
from WikiCatUtils import (
    scrape_page,
    read_categories,
    load_model_params,
    check_in_cache,
    save_model,
    ArticlePage,
    CategoryPage,
    Cache,
)



def get_fullpath(path):
    """ Check if the path given is absolute or relative, and if it's relative
    prepend the current directory """
    if path[0] != '/':
        return join(getcwd, path)
    else:
        return path

def download_category_page(page_uri, limit):


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action='store', help='Set\
                             verbosity level to one of [0 - 3], where 0 is\
                             quiet. Defaults to 1.', dest='verbosity', type=int, default=1)

    arg_parser.add_argument("categories_file", action='store', help='Path to\
                             CSV containing Category Page URIs, Max Amount of\
                             Articles (e.g. Philosophy, 100). Non-absolute paths are\
                            considered relative to the current directory.', dest='categories')

    arg_parser.add_argument("-r", "--representation", action='store', help='How to\
                            represent the data to the model. Options are BoW,\
                            GloVe, word2vec. Defaults to BoW', dest='repr', default='BoW')

    arg_parser.add_argument("-m", "--model", action='store', help='Path to JSON\
                            file containing a model / hyperparameter\
                            specification as outlined in ModelChoices.txt. Non-absolute paths are\
                            considered relative to the current directory.\
                            Defaults to Logistic Regression w/C=1.0', dest='model')

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


    args = arg_parser.parse_args()

    # Setup data
    cwd = getcwd()
    categories_to_download = read_categories(get_fullpath(args.categories))
    cache = Cache(get_fullpath(args.cache))
    for category_uri in categories_to_download:
        cache.loadCategory(category_uri, use_only_cache=args.cache_only)

    # Build training / validation / testing set.

    # Build representation of the data according to the arguments

    # Train model




