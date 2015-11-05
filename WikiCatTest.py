from __future__ import division, print_function # Python2 compat
from argparse import ArgumentParser
from os import getcwd, mkdir
from itertools import chain
from scipy.stats import entropy
import sys
import numpy as np
from os.path import (
    join,
    exists,
    getmtime,
)
from WikiCatUtils import (
    get_fullpath,
    read_categories,
    load_obj,
    ArticlePage,
    CategoryPage,
    Cache,
    Representer,
    Model,
)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-v", "--verbose", action='store', help='Set\
                             verbosity level to one of [0 - 3], where 0 is\
                             quiet. Defaults to 1.', dest='verbosity', type=int, default=1)

    arg_parser.add_argument("url", action='store', help='Valid (i.e. URLEncoded) article URL')

    """
    arg_parser.add_argument("-m", "--model-file", action='store', help='Path to\
                            where the trained model should be loaded from. Defaults\
                            to /tmp/WikiCat/model.pkl. Non-absolute paths are\
                            considered relative to the current directory.', dest='model_file',
                            default='/tmp/WikiCat/model.pkl')
    """

    arg_parser.add_argument("-r", "--root-dir", action='store', help='Path to\
                            where the representer, model, and cache are stored.\
                            Defaults to /tmp/WikiCat',
                            dest='root_dir', default='/tmp/WikiCat/')

    """
    arg_parser.add_argument("-c", "--cache", action='store', help="Directory containing the\
                            downloaded GloVe vector files.",
                            dest='cache', default='/tmp/WikiCat/cache')
    """
    args = arg_parser.parse_args()

    root_dir = get_fullpath(args.root_dir)
    model_fn = join(root_dir, 'model.pkl')
    repr_fn = join(root_dir, 'repr.pkl')
    if not (exists(model_fn) and exists(repr_fn)):
        print("Please run WikiCatBuild.py first and make sure that you use the "
              "same root directory")
        sys.exit(1)

    article = ArticlePage(args.url, verbosity=args.verbosity)
    model = load_obj(model_fn, verbosity=args.verbosity)
    embedder = load_obj(repr_fn, verbosity=args.verbosity)
    #embedder = Representer(get_fullpath(args.cache), verbosity=args.verbosity)
    article_rep = embedder.transform([article.Content()])
    model_predictions = model.predict_proba(article_rep).flatten()
    for ind, class_prob in enumerate(model_predictions):
        print("%s: %0.3f" % (model.ind_label_map[ind], class_prob))
    print("Final classification: %s" % (model.ind_label_map[np.argmax(model_predictions)]))

