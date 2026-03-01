
# %%
# Our sample training data
import collections
# Our sample training data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

print("Training Corpus:")
for doc in corpus:
    print(doc)
# %% [markdown]
# ## Step 2: Initialize Vocabulary and Pre-tokenize
#
# The BPE algorithm starts with a base vocabulary consisting of all unique characters present in the training data.
#
# We also need to pre-tokenize the corpus. This usually involves splitting the text into words (or word-like units) and then representing each word as a sequence of its individual characters. We often add a special end-of-word token (like `</w>`) to mark word boundaries, which helps the tokenizer learn subword units that align better with whole words.

# %%

# Step 2: Initialize Vocabulary and Pre-tokenize
unique_chars = set()
for doc in corpus:
    for char in doc:
        unique_chars.add(char)
vocab = list(unique_chars)
vocab.sort() # for consistent  ordering of characters making sure it is predictable
end_of_word ="</w>"
vocab.append(end_of_word) # add end of word token to the vocabulary
print("Initial vocabulary:", vocab)


# pre tokenize the corpus : Split the words and then characters
# we will split by space 
word_splits = {}
for doc in corpus:
    words = doc.split() # split by space
    for word in words:
        if word:
            char_list = list(word) + [end_of_word]
            # Use tuple for immutability of storing counts
            word_tuple = tuple(char_list)
            if word_tuple not in word_splits:
                word_splits[word_tuple] = 0
            word_splits[word_tuple] += 1
# %% [markdown]
# ### Helper Function: `get_pair_stats`
# helper function to get the frequency of the adjacent word
# %%
def get_pair_stats(splits):
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq # Add the frequency of the word to the pair count
    return pair_counts

# we want to take the most frequent pair and merge it into a new token
# %% [markdown]
# ### Helper Function: `merge_pair`
#
# This function takes a specific pair (`pair_to_merge`) that we want to combine and the current `splits`. It iterates through all the word representations in `splits`, replaces occurrences of the `pair_to_merge` with a new single token (concatenation of the pair), and returns the updated `splits`.
#
# **Input Example:**
# - `pair_to_merge`: `('i', 's')`
# - `splits`: `{('T', 'h', 'i', 's', '</w>'): 2, ('i', 's', '</w>'): 2, ...}`
#
# **Output Example (`new_splits`):**
# - `{('T', 'h', 'is', '</w>'): 2, ('is', '</w>'): 2, ...}` (assuming 'is' is the merged token)

# %%
def merge_pair(pair_to_merge, splits):
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second # create the new token by concatenating the pair
    for word_tuple , freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token) # add the merged token
                i += 2 # skip the next symbol since it's part of the merged pair
            else:
                new_symbols.append(symbols[i]) # add the current symbol
                i += 1
            new_splits[tuple(new_symbols)] = freq # update the new splits with the frequency
    return new_splits
# %% [markdown]
# ### Step 3: Iterative BPE Merging Loop
#
# Now we perform the core BPE training. We'll loop for a fixed number of merges (`num_merges`). In each iteration:
# 1. Calculate the frequencies of all adjacent pairs in the current word representations using `get_pair_stats`.
# 2. Find the pair with the highest frequency (`best_pair`).
# 3. Merge this `best_pair` across all word representations using `merge_pair`.
# 4. Add the newly formed token (concatenation of `best_pair`) to our vocabulary (`vocab`).
# 5. Store the merge rule (mapping the pair to the new token) in the `merges` dictionary.
#
# We'll add print statements to observe the state at each step of the loop.

# %%
# --- BPE Training Loop Initialization 
# ### Step 3: Iterative BPE Merging Loop
# %%
num_merges = 15
# Stores merge rules, e.g., {('a', 'b'): 'ab'}
# Example: {('T', 'h'): 'Th'}
merges = {}
# Initial word splits: {('T', 'h', 'i', 's', '</w>'): 2, ('i', 's', '</w>'): 2, ...}
current_splits = word_splits.copy()
print("\n--- Starting BPE Merges ---")
print(f"Initial Splits: {current_splits}")
print("-" * 30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")
    # 1. Calculate Pair Frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break
    # Optional: Print top 5 pairs for inspection
    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 Pair Frequencies: {sorted_pairs[:5]}")    
    # 2. Find the Most Frequent Pair
    # The 'max' function iterates over all key-value pairs in the 'pair_stats' dictionary
    # The 'key=pair_stats.get' tells 'max' to use the frequency (value) for comparison, not the pair (key) itself
    # This way, 'max' selects the pair with the highest frequency
    best_pair = max(pair_stats,key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Best Pair to Merge: {best_pair} with frequency {best_freq}")
    # 3. Merge the Best Pair
    current_splits = merge_pair(best_pair,current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into '{new_token}'")
    print(f"Splits after merge: {current_splits}")

    # 4. Update Vocabulary
    vocab.append(new_token)
    print(f"Updated Vocabulary: {vocab}")
     # 5. Store Merge Rule
    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")

    print("-" * 30)
# %% [markdown]
# ### Step 4: Review Final Results
#
# After the loop finishes, we can examine the final state:
# - The learned merge rules (`merges`).
# - The final representation of words after merges (`current_splits`).
# - The complete vocabulary (`vocab`) containing initial characters and learned subword tokens.

# %%
# --- BPE Merges Complete ---
print("\n--- BPE Merges Complete ---")
print(f"Final Vocabulary Size: {len(vocab)}")
print("\nLearned Merges (Pair -> New Token):")
# Pretty print merges
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")
print("\nFinal Word Representations after Merges:")
print(current_splits)
print("\nFinal Vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Use set to remove potential duplicates if any step introduced them
print(final_vocab_sorted)

# %%

