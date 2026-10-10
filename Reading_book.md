## One chart is better than hundreds words

-- Understand the branch
Key Concepts:

main Branch: Usually the trusted working version, and the first branch. (historically called master)
Feature Branch: A safe isolated space to develop without affecting the trusted version.
Merging: Combining changes from different branches.
![Branch Diagram](https://x.com/DeRonin_/article/2033587293064204349)


after the merge
![Branch Diagram](https://x.com/DeRonin_/article/2033587293064204349)

## LLM concept:
A vocabulary is created by retaining all unique words across both sentences
<img width="1339" height="658" alt="image" src="https://github.com/user-attachments/assets/d1bfa2ab-31c9-4b9c-811b-cf54f2a43a94" />

Using our vocabulary, we simply count how often a word in each sentence
appears, quite literally creating a bag of words. As a result, a bag-of-words
model aims to create representations of text in the form of numbers, also
called vectors or vector representations, observed in Figure 1-5. Throughout
the book, we refer to these kinds of models as representation models.

## How the Vector Values standardly work:
-1 (Present): A value of 1 means that word exists in the input sentence.Words present: is, cute, my, cat $\rightarrow$ mapped to [1, 1, 1, 1].   
-0 (Absent): A value of 0 means that word does not appear in the input sentence.Words absent: that, a, dog $\rightarrow$ mapped to [0, 0, 0].   
<img width="1341" height="737" alt="image" src="https://github.com/user-attachments/assets/b7a6a5e7-f797-4c2b-abc5-13ebda0aaa6a" />

## for tabular data processing
In tabular data, the input features are already numerical, so **vectorization means grouping these numerical values into a single row vector (or mathematical array) for each observation.**

Unlike text, where words must first be mapped to numerical IDs or word counts, numerical tabular data skips the text-to-number mapping step.

---

### 1. Vectorization of Numerical Features

If you have a table with continuous numerical columns (e.g., Age, Income, Credit Score), vectorizing simply means arranging those features into a $1 \times d$ feature vector for each instance (where $d$ is the number of features):

| Customer ID | Age ($x_1$) | Income ($x_2$) | Credit Score ($x_3$) |
| --- | --- | --- | --- |
| **User A** | 25 | 50,000 | 720 |
| **User B** | 42 | 85,000 | 680 |

The vectorized representation for **User A** is:


$$\mathbf{x}_{\text{User A}} = \begin{bmatrix} 25 & 50000 & 720 \end{bmatrix}$$

---

### 2. Standardizing & Normalizing Numerical Vectors

Even though numbers are directly readable by machine learning algorithms, raw numerical vectors often cause issues because features operate on completely different scales (e.g., Age ranges from $0\text{--}100$, while Income ranges from $0\text{--}100,000+$).

To prevent large-scale numbers from dominating gradient-based optimization or distance measurements, raw numerical vectors are typically transformed through **Scaling**:

* **Standard Scaling ($Z$-score Normalization):**
Centers features around mean $\mu=0$ with standard deviation $\sigma=1$:

$$x' = \frac{x - \mu}{\sigma}$$


* **Min-Max Scaling:**
Rescales values into a fixed bound (typically $[0, 1]$):

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$



---

### 3. What About Categorical Features in Tabular Data?

If a tabular feature is non-numerical (e.g., `City: ["Toronto", "Vaughan", "Ottawa"]`), it undergoes categorical vectorization before joining the final feature vector:

* **One-Hot Encoding:** Converts categorical variables into binary dummy vectors ($0$s and $1$s), similar to Bag-of-Words.
* `"Vaughan"` $\rightarrow \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}$


* **Target / Weight of Evidence (WoE) Encoding:** Replaces each categorical level with a scalar statistical value derived from the target variable distribution.

---

### Summary

For numerical tabular data:

1. **Raw Numerical Row** $\rightarrow$ Arrayed directly into a dense numerical vector $\mathbf{x} = [x_1, x_2, \dots, x_d]$.
2. **Preprocessing/Scaling** $\rightarrow$ Standardized to ensure all vector dimensions occupy comparable magnitudes.

## Embeddings are vector representations of data that attempt to capture its meaning
neural networks can have
many layers where each connection has a certain weight depending on the input. These weights are often referred to as the parameters of the model.The resulting embeddings capture the meaning of words but what exactly does that mean? To illustrate this phenomenon, let’s somewhat oversimplify and imagine we have embeddings of several words, namely “apple” and “baby.” Embeddings attempt to capture meaning by representing the properties of words. For instance, the word “baby” might score high on the properties “newborn” and “human” while the word “apple” scores low on these properties.
<img width="1102" height="742" alt="image" src="https://github.com/user-attachments/assets/95efb0c0-bf00-4178-a9c8-128c9e3fb890" />
<img width="1124" height="311" alt="image" src="https://github.com/user-attachments/assets/8232d912-953d-4a34-97f7-3fda6b47b770" />

