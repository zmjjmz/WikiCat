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
import re

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

def strip_unicode(s):
    return re.sub(r'[^\x00-\x7F]+','', s)
    return s.encode('unicode_escape').decode('unicode_escape')



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
            self.title = strip_unicode(self.page_soup.find(id='firstHeading').text)
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
        self.category = strip_unicode(category)

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
        return self._check_cached(join(articlePage.Category(), articlePage.Title()),
                                  articlePage.lastModified())

    def _save_article(self, category, article):
        article.loadAll()
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
            self._load_article(category, article_title)

    def loadCategory(self, category_uri, maxlinks=None, only_use_cached=False):
        category_page = CategoryPage(category_uri, verbosity=self.verbosity)
        if (not only_use_cached):
            # If no category exists in the folder or we're not only using cached
            # ones, we should download stuff. The caveat this comes with is
            if not exists(join(self.path, category_page.Category())):
                if self.verbosity > 1:
                    print("Creating directory for %s" %
                          category_page.Category())
                mkdir(strip_unicode(join(self.path, category_page.Category())))
            links = category_page.Content(maxlinks=maxlinks)
            articles = [ArticlePage(uri, verbosity=self.verbosity) for uri in
                        links]
            if self.verbosity > 1:
                print("Got %d articles for category %s" % (len(articles),
                                                           category_page.Category()))
            for article in articles:
                article.setCategory(category_page.Category())
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

def check_in_cache(cache_path, Page):
    """ Returns False if we need to (re-)download this page, True
    otherwise """
    title = Page.title
    if exists(join(cache_path, title)):
        # get the last modified time for the category directory
        cached_last_modified = getmtime(join(cache_path, title))
        if Page.last_modified > cached_last_modified:
            # this means that manually going in and editing the cache is a poor
            # decision, since this function will fail this check
            return False
        else:
            return True
    else:
        # if it's not in the cache, definitely need to download
        return False


