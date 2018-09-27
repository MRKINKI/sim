from .nonsence_word import NonSenceWord


class NgramTransform:
    def __init__(self):
        self.nonsencewords = NonSenceWord.nonsenceword

    def filter_word(self, text):
        if text.lower() in self.nonsencewords:
            return True
        return False

    def filter_ngram(self, gram, mode='any'):
        filtered = [self.filter_word(w) for w in gram]
        if mode == 'any':
            return any(filtered)
        elif mode == 'all':
            return all(filtered)
        elif mode == 'ends':
            return filtered[0] or filtered[-1]
        else:
            raise ValueError('Invalid mode: %s' % mode)

    def get_ngrams(self, words, n=1, filter_fn=True, as_strings=True):
        def _skip(gram):
            if filter_fn:
                return self.filter_ngram(gram)
            else:
                return False

        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]
        return ngrams
