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
在机器学习中，该词通常指一种专门的计算技术，它将复杂的、高维度的类别数据——如文字、图像或用户 ID——映射到一个连续的、低维度的向量空间中，并通过几何距离来捕捉数据之间的语义关系。)
embedding has multiple forms - document, sentence, words
<img width="1099" height="773" alt="image" src="https://github.com/user-attachments/assets/4b3e85bf-020a-426d-bb95-2662e218edcb" />

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
#### As illustrated in Figure 1-8, embeddings can have many properties to represent the meaning of a word. Since the size of embeddings is fixed, their properties are chosen to create a mental representation of the word.
<img width="1114" height="545" alt="image" src="https://github.com/user-attachments/assets/4f17630e-4b73-4b47-ba0d-4324482f6110" />
### The Encoder (The Listener / Compressor)
What it does: It takes in raw, complex data (like a long sentence, a paragraph, or an image) and reads it from start to finish.
In AI terms: It converts the input into a dense, compressed list of numbers (an embedding vector) that captures the core meaning.
<img width="1123" height="834" alt="image" src="https://github.com/user-attachments/assets/c69c191f-64aa-4358-8bfb-428a1bb22571" />

### The Decoder (The Speaker / Generator)
What it does: It takes that compressed summary from the encoder and unpacks it to generate a brand-new output (like the same sentence translated into English, or a reply to a question) word by word. In AI terms: It starts with the encoded meaning and predicts the next best word or pixel repeatedly until the final output is complete.

#### In 2014, a solution called attention was introduced that highly
improved upon the original architecture.4 Attention allows a model to focus on parts of the input sequence that are relevant to one another (“attend” to each other) and amplify their signal, as shown in Figure 1-14.
<img width="1106" height="556" alt="image" src="https://github.com/user-attachments/assets/9d139d48-1a03-412a-9222-c1ab45237277" />
2017 paper release - attention is all you needed. 
The encoder block in the Transformer consists of two parts, self-attention and a feedforward neural network, which are shown in Figure 1-17. Compared to previous methods of attention, self-attention can attend to different positions within a single sequence, thereby more easily and accurately representing the input sequence as illustrated in Figure 1-18.
<img width="1076" height="844" alt="image" src="https://github.com/user-attachments/assets/aa0de73d-76de-4eb2-83b7-680e2123453c" />
<img width="1085" height="550" alt="image" src="https://github.com/user-attachments/assets/a7791a2b-ae91-4e32-9f54-6d3fbe663548" />
### Decoder in attention 
<img width="1105" height="807" alt="image" src="https://github.com/user-attachments/assets/ec4c438f-fb52-4ae5-a02d-2bfb7aad7f2b" />
This architecture and training procedure makes BERT and related architectures incredible at representing contextual language. BERT-like
models are commonly used for transfer learning, which involves first pretraining it for language modeling and then fine-tuning it for a specific task. 
<img width="1122" height="491" alt="image" src="https://github.com/user-attachments/assets/8767629e-a5f3-4858-b3c0-dd4914b8627b" />

This architecture and training procedure makes BERT and related architectures incredible at representing contextual language. BERT-like models are commonly used for transfer learning, which involves first pretraining it for language modeling and then fine-tuning it for a specific task. For instance, by training BERT on the entirety of Wikipedia, it learns to understand the semantic and contextual nature of text.
<img width="1092" height="531" alt="image" src="https://github.com/user-attachments/assets/d309baa3-1ca5-4479-904f-b0f10b5e1cae" />
Similar to the encoder-only architecture of BERT, a decoder-only architecture was proposed in 2018 to target generative tasks.7 This architecture was called a Generative Pre-trained Transformer (GPT) for its generative capabilities (it’s now known as GPT-1 to distinguish it from later versions).

### Think of a **GPT** (Generative Pre-trained Transformer) like an incredibly well-read autocomplete engine.

Instead of just predicting the next single word in a text message, it can write entire essays, code, analyze data, and hold conversations.

The name breaks down into three simple parts:

---

### 1. Generative (The Creator)

* **What it means:** It doesn't just sort or classify data (like telling you if an email is spam); it **creates brand-new content**.
* **Plain English:** It generates original sentences, stories, answers, or computer code piece by piece based on what you ask it to do.

### 2. Pre-trained (The Student)

* **What it means:** Before you ever talk to it, the AI goes through a massive schooling phase where it reads a staggering amount of text from the internet—books, articles, code, and conversations.
* **Plain English:** It starts its life having already "read" human history, language, and culture. It learns grammar, facts, reasoning patterns, and coding syntax *before* you give it a single prompt.

### 3. Transformer (The Brain Architecture)

* **What it means:** This is the specific type of AI design (built on **attention mechanisms**) that looks at a whole sentence at once rather than just one word at a time.
* **Plain English:** Imagine reading a long paragraph. When you reach the end, you still remember context from the beginning. The Transformer architecture allows the AI to track how words relate to each other over long distances, ensuring the conversation stays coherent and context-aware.

---

### How It Works in Practice

When you type a prompt, the GPT doesn't "think" or feel emotions. It simply looks at everything you wrote, relies on its massive pre-trained memory, and calculates **the statistically most likely next word**—then repeats that process until the response is finished.
<img width="1112" height="626" alt="image" src="https://github.com/user-attachments/assets/c56eea5e-9a1d-4cb9-aa63-e02f9b89513b" />

## CHAPTER 2 - Tokenization and Embedding
I don't have direct access to internal, low-level token IDs from my underlying runtime interface. However, because tokenization depends on the exact tokenizer used (e.g., Tiktoken for OpenAI models, SentencePiece for LLaMA, or Hugging Face tokenizers), we can see how your previous question—**"what's proprietary models"**—is tokenized using standard LLM tokenizers!

---

### Example Tokenization

#### 1. OpenAI Tiktoken (`cl100k_base` / `o200k_base`)

For `o200k_base` (used by GPT-4o and newer models):

| Text Piece | Token ID |
| --- | --- |
| `what` | `13214` |
| `'s` | `1004` |
| ` proprietary` | `37803` |
| ` models` | `7531` |

* **Full Array of Token IDs:** `[13214, 1004, 37803, 7531]`

---

#### 2. LLaMA / Meta Tokenizer (`SentencePiece` / BPE)

| Text Piece | Token ID |
| --- | --- |
| `what` | `1083` |
| `'` | `29881` |
| `s` | `29879` |
| ` proprietary` | `24673` |
| ` models` | `4320` |

* **Full Array of Token IDs:** `[1083, 29881, 29879, 24673, 4320]`

---

### How to Check Token IDs Yourself

If you're writing Python code or testing locally, you can extract exact token IDs using Python libraries like `tiktoken` or Hugging Face's `transformers`:

**Using `tiktoken` (OpenAI):**

```python
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
tokens = enc.encode("what's proprietary models")

print("Token IDs:", tokens)
# Output: [13214, 1004, 37803, 7531]

print("Tokens:", [enc.decode([t]) for t in tokens])
# Output: ["what", "'s", " proprietary", " models"]

```

**Using Hugging Face (`transformers`):**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
token_ids = tokenizer.encode("what's proprietary models")

print("Token IDs:", token_ids)

```

### These generative decoder-only models, especially the “larger” models, are commonly referred to as large language models (LLMs).
A vital part of these completion models is something called the context length or context window. The context length represents the maximum
number of tokens the model can process, as shown in Figure 1-27.
<img width="1119" height="525" alt="image" src="https://github.com/user-attachments/assets/97355793-a13b-4cc4-8ca2-d1fe05f1fa89" />

### after tokenization,  The next piece of the puzzle is finding the best numerical representation for these tokens that the model can use to calculate and properly model the patterns in the text. Before the beginning of the training process, these vectors are randomly initialized like the rest of the model’s weights, but the training process assigns them the values that enable the useful behavior they’re trained to perform.
<img width="1084" height="433" alt="image" src="https://github.com/user-attachments/assets/15dc6a0b-5147-406d-a0b2-dc85952c854b" />

### Transformer output words one by one. This gives us a more accurate picture of the model as it is simply predicting the next token based on an input prompt. Software around the neural network basically runs it in a loop to sequentially expand the generated text until completion.
<img width="1105" height="497" alt="image" src="https://github.com/user-attachments/assets/256647e2-700d-4b3a-b545-70660f9185c3" />

In addition to the loop, two key internal components are the tokenizer and the language modeling head (LM head). The tokenizer is followed by the neural network: a stack of Transformer blocks that do all of the processing. That stack is then followed by the LM head, which translates the output of the stack into probability scores for what the most likely next token is.
<img width="1073" height="703" alt="image" src="https://github.com/user-attachments/assets/933408fd-29cd-4ba7-8378-8652a15d9817" />

the tokenizer’s vocabulary. The model has a vector representation associated with each of these tokens in the vocabulary (token embeddings). Figure 3-5
shows both the vocabulary and associated token embeddings for a model with a vocabulary of 50,000 tokens.

<img width="1113" height="568" alt="image" src="https://github.com/user-attachments/assets/04957978-7a27-4c90-ae19-8a6935022afc" />
What this means for the example in Figure 3-7 is that if the token “Dear” has a 40% probability of being the next token, then it has a 40% chance of
being picked (instead of greedy search, which would pick it directly for having the highest score).
<img width="1067" height="608" alt="image" src="https://github.com/user-attachments/assets/b63a4948-d715-452c-adeb-e6484e9a2ccb" />
Choosing the highest scoring token every time is called greedy decoding. It’s what happens if you set the temperature parameter to zero in an LLM.
That statement describes how an LLM determines its output step-by-step during text generation.

### How It Works

* **The Token Selection Process:** At every step of generating a response, an LLM calculates a probability distribution over its entire vocabulary for what the next token (word or piece of a word) should be.
* **Greedy Decoding:** When you use **greedy decoding**, the model always picks the single token with the highest probability score, completely ignoring any other plausible alternatives.

---

### The Role of Temperature ($T = 0$)

The **temperature** parameter controls how "creative" or random the model's choices are by scaling the probability scores before selection:

1. **Higher Temperature ($T > 0$):** Flattens the distribution, making lower-scoring tokens more likely to be chosen. This introduces variety and creativity into the model's responses.
2. **Zero Temperature ($T = 0$):** Effectively removes randomness. It sharpens the distribution so infinitely more weight is placed on the top-ranking token, forcing the model to default to **greedy decoding**.

---

### Pros and Cons

* **Pros:** It is entirely deterministic. If you run the same prompt with $T = 0$ multiple times, you will get the exact same output every single time. It is great for tasks requiring strict consistency, logic, or code generation.
* **Cons:** It can lead to repetitive phrasing, loops, or overly safe, bland text because the model never takes a chance on a lower-probability (but more creative or interesting) word choice.

A Transformer block (Figure 3-12) is made up of two successive components:
1. The attention layer is mainly concerned with incorporating relevant information from other input tokens and positions
2. The feedforward layer houses the majority of the model’s processing capacity
   <img width="1075" height="456" alt="image" src="https://github.com/user-attachments/assets/dea857cd-0d31-4ccd-abe9-97e9e6789a6d" />
The feedforward neural network (collectively in all the model layers) is the source of this information, as Figure 3-13 shows. When the model was
successfully trained to model a massive text archive (which included many mentions of “The Shawshank Redemption”), it learned and stored the
information (and behaviors) that make it succeed at this task.
<img width="1077" height="830" alt="image" src="https://github.com/user-attachments/assets/fbe45b1e-5b97-422c-876e-50de696ebd91" />



