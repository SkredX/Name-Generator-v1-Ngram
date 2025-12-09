# Understanding Markov Chains: The Bigram Model

A **Markov Chain** is a mathematical system that hops from one "state" (situation) to another based on probability.

In the context of Natural Language Processing (NLP), a **Bigram Model** is the simplest form of a Markov Chain. It is used to predict the next word in a sequence based *only* on the current word.

## 1. The Core Concept: The Markov Property
The defining characteristic of a Markov Chain is "memorylessness."

> **The Markov Property:** The probability of moving to the next state depends **only** on the current state and not on the sequence of events that preceded it.

Mathematically, for a sequence of words $w$, the probability of the next word $w_i$ is calculated as:

$$
P(w_i | w_{1}, w_{2}, ..., w_{i-1}) \approx P(w_i | w_{i-1})
$$

This means if we want to predict what comes after "Apple," we look *only* at "Apple," ignoring the 50 words written before it.


## 2. How a Bigram Model is Built

To create a Bigram model, we follow three steps: **Tokenize**, **Pair**, and **Calculate Probability**.

### Step A: The Corpus (Training Data)
Let's use a very simple dataset (corpus):
`"I like tea. I like coffee."`

### Step B: Create Bigrams
We break the text into pairs of consecutive words (Bigrams).

* (Start, I)
* (I, like)
* (like, tea)
* (tea, End)
* (Start, I)
* (I, like)
* (like, coffee)
* (coffee, End)

### Step C: Calculate Transition Probabilities
We count how often word B follows word A and convert that into a percentage.

| Current State (Word A) | Possible Next States (Word B) | Probability | Calculation |
| :--- | :--- | :--- | :--- |
| **I** | like | **100%** | "like" follows "I" 2 out of 2 times. |
| **like** | tea | **50%** | "tea" follows "like" 1 out of 2 times. |
| **like** | coffee | **50%** | "coffee" follows "like" 1 out of 2 times. |
| **tea** | . (End) | **100%** | End follows "tea" 1 out of 1 time. |


## 3. Generating Text
Once the model is trained, generating text is simply a "random walk" through the states.

1.  **Start:** The model looks for words that start a sentence. In our data, it is always **"I"**.
2.  **Current State "I":** The model looks at its table. The only option is **"like"**.
3.  **Current State "like":** The model flips a coin (50/50 chance).
    * *Heads:* It picks **"tea"**.
4.  **Current State "tea":** The model looks at its table. The next step is **End**.

**Result:** "I like tea."

If we ran it again and the coin flip was different, we might generate "I like coffee."

## 4. Limitations
While Bigram models are efficient, they lack **context**. Because they only look one word back, they often produce sentences that are grammatically correct locally but nonsensical globally.

* **Example:** "The dog went to the store to buy a car drove fast."
* **Why:** The link `(buy, a)` is valid, and `(a, car)` is valid, and `(car, drove)` is valid. But the model forgot that "The dog" was the subject at the start of the sentence.
