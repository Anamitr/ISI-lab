import time


def anagrams(word):
    """ Generate all of the anagrams of a word. """
    if len(word) < 2:
        yield word
    else:
        for i, letter in enumerate(word):
            if not letter in word[:i]:  # avoid duplicating earlier words
                for j in anagrams(word[:i] + word[i + 1:]):
                    yield j + letter


if __name__ == "__main__":
    start = time.time()
    all_permutations = []
    for i in anagrams(''.join(['p', ' ', ' ', ' ', 'g', ' ', ' ', ' ', '*', ' ', ' ', ' ', ' ', ' ', ' '])):
        print(i)
        all_permutations.append(i)
    end = time.time()
    elapsed = end - start
    print("Time:", elapsed)


def permutations_with_partial_repetitions(list_to_permutate : list):
    all_permutations = []
    for i in anagrams(''.join(list_to_permutate)):
        all_permutations.append(i)
    return all_permutations