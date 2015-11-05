from __future__ import division, print_function # Python2 compat
from argparse import ArgumentParser
from os import getcwd, mkdir
from itertools import chain
from scipy.stats import entropy
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

    arg_parser.add_argument("-m", "--model-file", action='store', help='Path to\
                            where the trained model should be loaded from. Defaults\
                            to /tmp/WikiCat/model.pkl. Non-absolute paths are\
                            considered relative to the current directory.', dest='model_file',
                            default='/tmp/WikiCat/model.pkl')

    arg_parser.add_argument("-r", "--repr-file", action='store', help='Path to\
                            where the representer (i.e. normalization params,\
                            vocab) object is stored.', dest='repr_file',
                            default='/tmp/WikiCat/repr.pkl')

    arg_parser.add_argument("-c", "--cache", action='store', help="Directory containing the\
                            downloaded GloVe vector files.",
                            dest='cache', default='/tmp/WikiCat/cache')

    args = arg_parser.parse_args()
    article = ArticlePage(args.url, verbosity=args.verbosity)
    model = load_obj(get_fullpath(args.model_file), verbosity=args.verbosity)
    embedder = load_obj(get_fullpath(args.repr_file), verbosity=args.verbosity)
    #embedder = Representer(get_fullpath(args.cache), verbosity=args.verbosity)
    article_rep = embedder.transform([article.Content()])
    #print(article_rep)
    model_predictions = model.predict_proba(article_rep).flatten()
    for ind, class_prob in enumerate(model_predictions):
        print("%s: %0.3f" % (model.ind_label_map[ind], class_prob))
    print("Final classification: %s" % (model.ind_label_map[np.argmax(model_predictions)]))

