import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # A page that has no links at all should be interpreted as having one link for every page in the corpus
    corrected_corpus = {k: (v if v else set(corpus.keys())) for k, v in corpus.items()}

    model = dict.fromkeys(corrected_corpus.keys(), (1 - damping_factor) / len(corrected_corpus))
    for link in corpus[page]:
        model[link] += damping_factor / len(corpus[page])
    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    first = random.choice(list(corpus))
    model = transition_model(corpus, first, damping_factor)
    pagerank[first] = 1

    for _ in range(n-1):
        next = random.choices(list(model.keys()), list(model.values()))[0]
        pagerank[next] = pagerank.get(next, 0) + 1
        model = transition_model(corpus, next, damping_factor)
    
    pagerank = {key: value / n for key, value in pagerank.items()}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    old_pagerank = {}
    new_pagerank = dict.fromkeys(corpus.keys(), 1 / len(corpus))
    # A page that has no links at all should be interpreted as having one link for every page in the corpus
    corrected_corpus = {k: (v if v else set(corpus.keys())) for k, v in corpus.items()}

    while not close(old_pagerank, new_pagerank):
        old_pagerank = new_pagerank.copy()

        for page in old_pagerank:
            link_sum = sum(old_pagerank[link] / len(corrected_corpus[link]) for link in
                           corrected_corpus if page in corrected_corpus[link])
            new_pagerank[page] = ((1 - damping_factor) /
                                  len(corrected_corpus) + damping_factor * link_sum)

    return new_pagerank


def close(old, new):
    if not old:
        return False
    return all(abs(old[page] - new[page]) < 0.001 for page in old)


if __name__ == "__main__":
    main()

