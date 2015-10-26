from os.path import (
    exists,
    join,
    getmtime,
)
from os import mkdir, listdir
import pickle


class WikiPage:
    def __init__(self, uri):
        # Download the page
        # Parse it with BeautifulSoup
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
            pass
        else:
            return self.last_modified

    def Title(self):
        if self.title is None:
            # figure out the title
            pass
        else:
            return self.title

    def Category(self):
        raise NotImplementedError

    def Content(self):
        raise NotImplementedError

    def loadAll(self):
        # since generally we leave everything to be lazily evaluated, loadAll
        # will evaluate everything so this object can be pickled and reused
        self.lastModified()
        self.Title()
        self.Category()
        self.Content()

class ArticlePage(WikiPage):
    def Content(self):
        if self.content is None:
            # figure out the content by scraping
        else:
            return self.content

    def Category(self):
        if self.category is None:
            # We can't figure out the category from the Wiki page, as an article
            # can be listed in many categories
            raise ValueError
        else:
            return self.category

    def setCategory(self, category):
        self.category = category

class CategoryPage(WikiPage):
    def Content(self):
        if self.content is None:
            # return the list of links making up the articles
            # note that some categories won't show all of the
        else:
            return self.content

    def Category(self):
        return self.Title()


class Cache:
    """ This object handles the storage and retrieval of Wikipedia articles and
    category pages"""
    def __init__(self, path):
        if not exists(path):
            mkdir(path)
        self.path = path
        self.contents = {} # map of article title to article object

    def _check_cached(self, page_path, pageLastModified):
        if exists(join(self.path, page_path)):
            cached_last_modified = getmtime(join(self.path, page_path))
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
        return self._check_cached(categoryPage.title, categoryPage.last_modified)

    def articleCached(self, articlePage):
        return self._check_cached(join(articlePage.category, articlePage.title),
                                  articlePage.last_modified)

    def _save_article(self, category, article):
        article.loadAll()
        self.contents[article.Title()] = article
        with open(join(join(self.path, category), article.Title()), 'w') as f:
                # no this isn't the absolute best way to serialize stuff
                pickle.dump(article, f)

    def _load_article(self, category, article_title):
        with open(join(join(self.path, category), article_title), 'r') as f:
            self.contents[article_title] = pickle.load(f)


    # convenience
    def _save_articles(self, category, articles):
        article.setCategory(category)
        for article in articles:
            self._save_article(category, article)

    def _load_articles(self, category, article_titles):
        for article_title in article_titles:
            self._load_article(category, article_title)




    def loadCategory(self, category_uri, only_use_cached=False):
        category_page = CategoryPage(category_uri)
        if not exists(join(self.path, category_page.Category())):
            mkdir(join(self.path, category_page.Category()))
        if only_use_cached:
            article_files = listdir(join(self.path, category_page.Category()))
            if self.verbosity > 0:
                if len(article_files) == 0:
                    print("WARNING: Category %s has no articles cached, turn\
                          off only-use-cached to get them")
            self._load_articles(category_page.Category(), article_files)
        else:
            articles = [ArticlePage(uri) for uri in category_page.Content()]
            for article in articles:
                if self.articleCached(article):
                    self._load_article(category_page.Category(),
                                       article.Title())
                else:
                    # nomenclature is a bit confusing, but for the purpose of
                    # getting everything in self.contents this is what we want
                    self._save_article(category_page.Category(),
                                       article.Title())

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


