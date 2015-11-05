from os.path import (
    exists,
    join,
    getmtime,
)
import random
from os import mkdir, listdir, getcwd
import pickle
from bs4 import BeautifulSoup as BS
import requests
from itertools import chain
from datetime import datetime
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils import resample, shuffle
from sklearn.cross_validation import cross_val_score
import re
import codecs
import urllib
from zipfile import ZipFile
from string import punctuation
import sys

def make_link(partial_link):
    # TODO: make this robust to being given a full link
    return ("https://en.wikipedia.org" + partial_link)

def get_fullpath(path):
    """ Check if the path given is absolute or relative, and if it's relative
    prepend the current directory """
    if path[0] != '/':
        return join(getcwd(), path)
    else:
        return path

def read_categories(filepath):
    """ Reads a list of line separated category_uris"""
    with open(filepath, 'r') as f:
        # Hopefully there aren't any newlines in the URLs
        return f.read().rstrip().split('\n')

def safe_path(s):
    no_unicode = re.sub(r'[^\x00-\x7F]+','', s)
    no_slashes = no_unicode.replace('/','_')
    return no_slashes

def resample_to_equal(labels):
    # just return the indices to be used
    split_by_label = {label:np.where(np.argmax(labels, axis=1) == label)[0]
                      for label in range(labels.shape[1])}
    min_size = min([len(idxs) for idxs in split_by_label.values()])
    all_data = []
    for label in split_by_label:
        split_by_label[label] = resample(split_by_label[label],
                                         n_samples=min_size, replace=False)
        these_labels = [label]*min_size
        all_data = chain(all_data, zip(split_by_label[label], these_labels))
    all_data = map(np.array,zip(*shuffle(list(all_data))))
    return all_data

def save_obj(obj, save_path, verbosity=1):
    if verbosity > 0:
        print("Saving object to %s" % save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(load_path, verbosity=1):
    if verbosity > 0:
        print("Loading model from %s" % load_path)
    with open(load_path, 'rb') as f:
        return pickle.load(f)

class WikiPage:
    def __init__(self, uri, verbosity=1):
        self.verbosity = verbosity
        # Download the page
        if self.verbosity > 2:
            print("Requesting %s" % uri)
        req = requests.get(uri)
        # Parse it with BeautifulSoup
        if self.verbosity > 2:
            print("Putting %s through BeautifulSoup" % uri)
        self.page_soup = BS(req.content, 'html.parser')
        # The data model for each page is simple -- a title (so we can store
        # them), a date modified (for the cache), the content (the stripped
        # text), and the category (a category page's category will be the same
        # as its title)
        self.last_modified = None
        self.title = None
        self.content = None
        self.category = None

    def lastModified(self):
        if self.last_modified is None:
            # figure out the last modified w/BeautifulSoup
            last_mod_text = self.page_soup.find(id='footer-info-lastmod').text
            # fragile af
            last_mod_date = last_mod_text.split('on ')[-1].split(', at')[0]
            self.last_modified = datetime.strptime(last_mod_date, '%d %B %Y').timestamp()
        return self.last_modified

    def Title(self):
        if self.title is None:
            self.title = safe_path(self.page_soup.find(id='firstHeading').text)
        return self.title

    def Category(self):
        raise NotImplementedError

    def Content(self):
        raise NotImplementedError

    def loadAll(self):
        # since generally we leave everything to be lazily evaluated, loadAll
        # will evaluate everything so this object can be pickled and reused
        if self.verbosity > 2:
            print("%s: Loading all" % self.Title())
        self.lastModified()
        self.Title()
        self.Category()
        self.Content()

class ArticlePage(WikiPage):
    def Content(self):
        if self.content is None:
            # figure out the content by scraping
            text_before_ref = self.page_soup.find(id='mw-content-text').get_text().split('References[edit]')[0]
            self.content = re.sub('\[(?:\d*)\]|\[(?:edit)\]','', text_before_ref)
        return self.content

    def Category(self):
        if self.category is None:
            # We can't figure out the category from the Wiki page, as an article
            # can be listed in many categories
            raise NotImplementedError("It is not possible to tell the category"
                                      " from the article, it must be set with"
                                      " setCategory()")
        else:
            return self.category

    def setCategory(self, category):
        self.category = safe_path(category)

class CategoryPage(WikiPage):
    def get_links(self, category_soup):
        categories = category_soup.find(id="mw-pages").find_all(class_="mw-category-group")
        links = list(chain(*[grp.find_all('a') for grp in categories]))
        links = [make_link(link.attrs['href']) for link in links]
        if self.verbosity > 1:
            print("%s: Found %d links" % (self.Title(), len(links)))
        return links

    def get_next_page(self, category_soup):
        next_links = list(filter(lambda x: x.text == "next page",
                        category_soup.find(id="mw-pages").find_all('a')))
        if len(next_links) == 0:
            if self.verbosity > 1:
                print("%s: Found last page" % self.Title())
            return None
        else:
            return make_link(next_links[0].attrs['href'])

    def Content(self, maxlinks=None):
        if self.content is None:
            # return the list of links making up the articles
            # note that some categories won't show all of the
            self.content = []
            page_counter = 0
            cur_soup = self.page_soup
            while (True):
                page_counter += 1
                if self.verbosity > 1:
                    print("%s: Page counter: %d" % (self.Title(), page_counter))
                self.content = list(chain(self.content, self.get_links(cur_soup)))
                """
                if (maxlinks is not None) and len(self.content) >= maxlinks:
                        self.content = self.content[:maxlinks]
                        break
                """
                next_page = self.get_next_page(cur_soup)
                if next_page is None:
                    break
                else:
                    req = requests.get(next_page)
                    cur_soup = BS(req.content, 'html.parser')
        random.shuffle(self.content)
        if maxlinks is not None:
            self.content = self.content[:maxlinks]
        return self.content

    def Category(self):
        return self.Title()


class Cache:
    """ This object handles the storage and retrieval of Wikipedia articles and
    category pages"""
    def __init__(self, path, verbosity=1):
        if not exists(path):
            mkdir(path)
        self.path = path
        self.verbosity = verbosity
        self.contents = {} # map of article title to article object

    def _check_cached(self, page_path, pageLastModified):
        if exists(join(self.path, page_path)):
            cached_last_modified = getmtime(join(self.path, page_path))
            if self.verbosity > 2:
                print("Cached last modified for %s: %0.2f\nWiki last modified"
                      "for %s: %0.2f" % (page_path, cached_last_modified,
                                         page_path, pageLastModified))
            if pageLastModified > cached_last_modified:
                return False
            else:
                return True
        else:
            return False

    def categoryCached(self, categoryPage):
        # this is a little broken since if any articles inside the category
        # folder get updated, then so does the date modified time on the folder,
        # however this would not reflect something like pages being added to the
        # category. This is fine as long as individual articles aren't updated
        # without the category being updated, but since each time we run this we
        # check the category page first we should be clear.
        return self._check_cached(categoryPage.Title(), categoryPage.lastModified())

    def articleCached(self, articlePage):
        try:
            return self._check_cached(join(articlePage.Category(), articlePage.Title()),
                                      articlePage.lastModified())
        except UnicodeDecodeError as ude:
            print("Could not check on %s because of unicode madness"
                    % join(articlePage.Category(), articlePage.Title()))
            return False

    def _save_article(self, category, article):
        article.loadAll()
        if article.Title() in self.contents:
            raise ValueError("Article %s already exists in Cache, this should not happen"
                             " unless an article is referenced by two"
                             " categories" % article.Title())
        self.contents[article.Title()] = article
        with open(join(join(self.path, category), article.Title()), 'wb') as f:
            # no this isn't the absolute best way to serialize stuff
            if self.verbosity > 1:
                print("Saving %s into category %s" % (article.Title(), category))
            page_soup_tmp = article.page_soup # can't pickle entire BS4 trees
            del article.page_soup
            pickle.dump(article, f)
            article.page_soup = page_soup_tmp

    def _load_article(self, category, article_title):
        if article_title in self.contents:
            raise ValueError("Article %s already exists in Cache, this should not happen"
                             " unless an article is referenced by two"
                             " categories" % article_title)
        with open(join(join(self.path, category), article_title), 'rb') as f:
            if self.verbosity > 1:
                print("Loading %s from category %s" % (article_title, category))
            self.contents[article_title] = pickle.load(f)


    # convenience
    def _save_articles(self, category, articles):
        article.setCategory(category)
        for article in articles:
            self._save_article(category, article)

    def _load_articles(self, category, article_titles):
        for article_title in article_titles:
            try:
                self._load_article(category, article_title)
            except ValueError as vae:
                if self.verbosity > 1:
                    print(vae)
                # We'll default to taking the article's first category (in
                # the order of the list of categories given)
                continue


    def loadCategory(self, category_uri, maxlinks=None, only_use_cached=False):
        category_page = CategoryPage(category_uri, verbosity=self.verbosity)
        if self.verbosity > 0:
            print("Loading category %s" % category_page.Category())
        if (not only_use_cached):
            # If no category exists in the folder or we're not only using cached
            # ones, we should download stuff. The caveat this comes with is
            if not exists(join(self.path, category_page.Category())):
                if self.verbosity > 1:
                    print("Creating directory for %s" %
                          category_page.Category())
                mkdir(join(self.path, category_page.Category()))
            links = category_page.Content(maxlinks=maxlinks)
            articles = [ArticlePage(uri, verbosity=self.verbosity) for uri in
                        links]
            if self.verbosity > 1:
                print("Got %d articles for category %s" % (len(articles),
                                                           category_page.Category()))
            for article in articles:
                article.setCategory(category_page.Category())
                try:
                    if self.articleCached(article):
                        if self.verbosity > 2:
                            print("Using cached version of %s" % article.Title())
                        self._load_article(category_page.Category(),
                                           article.Title())
                    else:
                        # nomenclature is a bit confusing, but for the purpose of
                        # getting everything in self.contents this is what we want
                        if self.verbosity > 2:
                            print("Downloading new version of %s" % article.Title())
                        self._save_article(category_page.Category(),
                                           article)
                except ValueError as vae:
                    if self.verbosity > 0:
                        print(vae)
                    # We'll default to taking the article's first category (in
                    # the order of the list of categories given)
                    continue

        else:
            if not exists(join(self.path, category_page.Category())):
                print("Category %s doesn't exist in the cache, skipping." %
                       category_page.Category())
                return

            article_files = listdir(join(self.path, category_page.Category()))
            if self.verbosity > 0:
                if len(article_files) == 0:
                    print("WARNING: %s has no articles cached, turn"
                          "off only-use-cached to get them" %
                          category_page.Category())
            self._load_articles(category_page.Category(), article_files)

    def get_dataset(self, train_perc, val_perc, test_perc):
        slice_names = ['train', 'val', 'test']
        if abs((train_perc + val_perc + test_perc) -  1.0) > 1e-7:
            if self.verbosity > 0:
                print("Train, Validation, and Test percentages must add up to 1!")
            raise ValueError
        perc_slices = {
            'train': (0, train_perc),
            'val': (train_perc, train_perc+val_perc),
            'test': (train_perc+val_perc, 1),
        }
        adjust_slice = lambda x, length: slice(int(x[0]*length), int(x[1]*length))

        if len(self.contents) == 0:
            if self.verbosity > 0:
                print("Make sure you've loaded some articles first!")
            raise ValueError
        category_article_map = {}
        for articleTitle in self.contents:
            article = self.contents[articleTitle]
            if article.Category() in category_article_map:
                # we only care about the content
                category_article_map[article.Category()].append(article.Content())
            else:
                category_article_map[article.Category()] = [article.Content()]

        category_label_map = {category:ind for ind, category in
                          enumerate(category_article_map.keys())}
        dset = {
            'train':[],
            'val':[],
            'test':[],
        }
        for category in category_article_map:
            random.shuffle(category_article_map[category])
            for split in slice_names:
                split_slice = adjust_slice(perc_slices[split],
                                           len(category_article_map[category]))
                sliced_cat = category_article_map[category][split_slice]
                sliced_labels = [category_label_map[category]] * len(category_article_map[category])
                dset[split] = chain(dset[split], zip(sliced_cat, sliced_labels))

        for split in slice_names:
            dset[split] = list(dset[split])
            random.shuffle(dset[split])
            dset[split] = list(zip(*dset[split]))

        return dset, category_label_map

class Model:
    def __init__(self, label_map):
        """ A simple wrapper around LogisticRegression """
        self.model = LogisticRegression(C=0.125, class_weight='auto')
        self.ind_label_map = {v:k for k, v in label_map.items()}

    def fit(self, train_dataset):
        self.model.fit(*train_dataset)
        return self

    def evaluate(self, dset):
        """ Run a full evaluation of the model given training, validation, and test sets"""
        train_score = self.score(dset['train'])
        val_score = self.score(dset['val'])
        cv_scores = cross_val_score(self.model, *dset['train'], cv=10)
        test_score = self.score(dset['test'])
        print("Train\tVal\tTest\tCV\n"
              "%0.2f\t%0.2f\t%0.2f\t%0.2f (+/- %0.2f)" % (train_score,
                                                          val_score, test_score,
                                                          np.average(cv_scores),
                                                          np.std(cv_scores)/2))
    def predict_proba(self, dataset):
        return self.model.predict_proba(dataset)

    def predict(self, dataset):
        return self.model.predict(dataset)

    def score(self, dataset):
        return self.model.score(*dataset)


def glove_read(dimsize, path, verbosity=1, existing=None):
    member = 'glove.6B.%dd.txt' % dimsize
    fn = join(path, 'glove_files/%s' % member)
    zipfn = join(path, 'glove_files/glove_6B.zip')
    if not exists(fn):
        if not exists(join(path, 'glove_files')):
            mkdir(join(path, 'glove_files'))
        if verbosity > 0:
            print("GloVe vectors of size %d aren't there" % dimsize)
            url = "http://nlp.stanford.edu/data/glove.6B.zip"
            print("Download GloVe vectors from %s to %s? y/n" % (url, zipfn))
            confirm = input().rstrip()
        else:
            confirm = 'y'
        if confirm != "y":
            print("Can't continue")
            sys.exit(1)
        else:
            try:
                urllib.request.urlretrieve(url, zipfn)
                if verbosity > 0:
                    print("Done downloading. Extracting...")
                archive = ZipFile(zipfn)
                archive.extractall(join(path, 'glove_files'))
                if verbosity > 0:
                    print("Done extracting")
            except:
                e = sys.exc_info()[0]
                print("Can't continue: %s" % e)
                sys.exit(1)
    vocab_ret = {}
    if verbosity > 0:
        #400000 is the size of the vocabulary
        print("Reading 400000 %d-dimensional vectors" % dimsize)
    with codecs.open(fn, 'r', 'utf8') as f:
        for line in f:
            split_line = str(line).rstrip().split(' ')
            word = split_line[0]
            if existing is not None and word not in existing:
                continue
            vocab_ret[word] = np.array(list(map(float, split_line[1:])))
    return vocab_ret

punct_matcher = re.compile(r'[{}]+'.format(re.escape(punctuation)))
space_matcher = re.compile(r'\s+')
def tokenize(article):
    stripped = article.rstrip().lstrip()
    no_punct = re.sub(punct_matcher, '', stripped)
    tokenized = re.split(space_matcher, no_punct)
    # While the wikipedia articles may impart information by their case,
    # the vectors we're using are all lowercase
    lowered = list(map(lambda x: x.lower(), tokenized))
    return lowered

class Representer:
    def __init__(self, cache_path, normalize=True,
                 verbosity=1, filter_by=None):
        """ Load up 200 dimensional Glove vectors from cache """
        self.dim = 200
        self.verbosity = verbosity
        self.normalize = normalize
        self.maxes = None
        self.mins = None
        # Normally I would do this in fit() but due to memory constraints
        # it should be done in the __init__
        if filter_by is not None:
            self.existing = set()
            for article in filter_by:
                for word in tokenize(article):
                    if word not in self.existing:
                        self.existing.add(word)
        else:
            self.existing = None
        self.vocab = glove_read(self.dim, cache_path, verbosity=verbosity,
                                existing=self.existing)
        if self.verbosity > 0:
            print("Vocabulary size: %d" % len(self.vocab))

    def fit(self, dataset):
        """ Figure out normalization parameters """
        # awkwardly requires that we transform first, so we need to turn off
        # normalization so it won't yell at us
        if self.verbosity > 0:
            print("Determining normalization constants")
        if self.normalize:
            self.normalize = False
            to_normalize = self.transform(dataset)
            self.normalize = True
            self.maxes = np.max(to_normalize, axis=0)
            self.mins = np.min(to_normalize, axis=0)
            if self.verbosity > 1:
                print("Maxes: %r\n Mins: %r" % (self.maxes, self.mins))
        return self

    def transform(self, dataset):
        vecs = []
        oov = 0
        total_words = 0
        if self.verbosity > 0:
            print("Transforming %d articles" % len(dataset))
        for article in dataset:
            avg_vec = np.zeros((1, self.dim))
            tok_unfilt = tokenize(article)
            tokenized_article = list(filter(lambda x: x in self.vocab,
                                            tok_unfilt))
            oov += len(tok_unfilt) - len(tokenized_article)
            total_words += len(tok_unfilt)

            for word in tokenized_article:
                avg_vec += (1/len(tokenized_article))*(self.vocab[word].reshape(1,self.dim))
            vecs.append(avg_vec)
        if self.verbosity > 0:
            print("%d words were OOV out of %d total" % (oov, total_words))
        vecs = np.vstack(vecs)
        if self.normalize:
            vecs = (vecs - self.maxes) / (self.maxes - self.mins)
            vecs[np.where(np.isnan(vecs))] = 0.5
        return vecs

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

