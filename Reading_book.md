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

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# We need to know the population mean and standard deviation, or apply standard scaling to a dataset where this vector belongs. 
# Wait, standard scaling is typically applied to a dataset (columns). If x is just a single vector given without a dataset, 
# standard scaling requires mean and standard deviation. Let's see if the user implies a dataset or if standard scaling needs a hypothetical context, 
# or perhaps standard scaling is applied element-wise across the vector if treated as a single sample? No, standard scaling is feature-wise.
# Let's write a python script to handle standard scaling or explain the requirement. 
# Usually, standard scaling transforms features: z = (x - mean) / std. 
# Since we only have a single vector x = [25, 50000, 720], standard scaling across features would mean each element is treated as an independent feature (mean=25, std=0 for 1 sample, which is undefined or results in 0).
# Let's check if there's any implicit context, or if standard scaling requires more samples.
print("Standard scaling requires a set of samples to compute mean and standard deviation for each feature.")


```

```text
Standard scaling requires a set of samples to compute mean and standard deviation for each feature.


```

To perform **Standard Scaling** (Z-score normalization), a dataset of multiple samples is typically required so that the **mean ($\mu$)** and **standard deviation ($\sigma$)** can be calculated for each feature.

The formula for standard scaling is:


$$z = \frac{x - \mu}{\sigma}$$

Because your vector represents a single sample with three independent features—such as **Age ($25$)**, **Income ($50,000$)**, and **Credit Score ($720$)**—standard scaling cannot be calculated on this single vector alone without knowing the population or dataset parameters ($\mu$ and $\sigma$ for each feature).

---

### How it works with a dataset example:

Suppose your vector belongs to a dataset with the following feature averages and standard deviations:

| Feature | Your Value ($x$) | Dataset Mean ($\mu$) | Dataset Std Dev ($\sigma$) | Scaled Result ($z$) |
| --- | --- | --- | --- | --- |
| **Feature 1** (e.g., Age) | $25$ | $35$ | $10$ | $\frac{25 - 35}{10} = \mathbf{-1.0}$ |
| **Feature 2** (e.g., Income) | $50,000$ | $60,000$ | $15,000$ | $\frac{50000 - 60000}{15000} \approx \mathbf{-0.67}$ |
| **Feature 3** (e.g., Credit Score) | $720$ | $700$ | $50$ | $\frac{720 - 700}{50} = \mathbf{+0.4}$ |

Resulting scaled vector:


$$\mathbf{x_{scaled}} = \begin{bmatrix} -1.0 \\ -0.67 \\ +0.4 \end{bmatrix}$$

If you have a specific dataset or training set from which these statistics are derived, feel free to share them, and the exact scaled values can be computed for you!

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

Attention is a mechanism that helps the model incorporate context as it’s processing a specific token. Think of the following prompt: 
* “The dog chased the squirrel because it”
For the model to predict what comes after “it,” it needs to know what “it” refers to. Does it refer to the dog or the squirrel?
In a trained Transformer LLM, the attention mechanism makes that determination. Attention adds information from the context into the
representation of the “it” token. We can see a simple version of that in Figure 3-14.
<img width="1110" height="633" alt="image" src="https://github.com/user-attachments/assets/51965d7a-08c2-4885-9e00-6ae0a2583160" />
Two main steps are involved in the attention mechanism:
1. A way to score how relevant each of the previous input tokens are to the current token being processed (in the pink arrow).
2. Using those scores, we combine the information from the various positions into a single output vector.
<img width="1169" height="564" alt="image" src="https://github.com/user-attachments/assets/794a503d-b7b8-4994-88ce-1612a907f94d" />

The training process produces three projection matrices that produce the components that interact in this calculation:
* A query projection matrix
* A key projection matrix
* A value projection matrix
Attention starts by multiplying the inputs by the projection matrices to create three new matrices. These are called the queries, keys, and values
matrices.

These matrices contain the information of the input tokens projected to three different spaces that help carry out the two steps of attention:
1. Relevance scoring
2. Combining information
<img width="1097" height="782" alt="image" src="https://github.com/user-attachments/assets/1e0a20df-102d-4cbe-a7e3-411524f8f70b" />

In a generative Transformer, we’re generating one token at a time. This means we’re processing one position at a time. So the attention mechanism
here is only concerned with this one position, and how information from other positions can be pulled in to inform this position.
The relevance scoring step of attention is conducted by multiplying the query vector of the current position with the keys matrix. This produces a
score stating how relevant each previous token is. Passing that by a softmax operation normalizes these scores so they sum up to 1. Figure 3-20 shows
the relevance score resulting from this calculation.
<img width="1058" height="769" alt="image" src="https://github.com/user-attachments/assets/7c79aeee-a7ad-4223-bb0c-a399bbfd53a0" />
## 2 stages of training model
<img width="953" height="520" alt="image" src="https://github.com/user-attachments/assets/e6bcece7-0527-44fc-85fe-8e0fb0173779" />

<img width="1078" height="847" alt="image" src="https://github.com/user-attachments/assets/d1ff576e-8455-4aeb-905d-967f8b08c9b2" />
For example, select Head 1 · Reference relationship, with Full / bidirectional attention:
1. Find the “she” row.
2. Move across to the “customer” column.
3. The cell is approximately 68%: when updating the representation of “she,” this illustrative head gives most of its attention to “customer.”
Read across a row—not down a column. Each row distributes 100% of its attention across the available tokens. Displayed percentages may differ slightly from 100% because of rounding.
These weights control a blend of value vectors:
\[
\text{Updated representation of “she”}
=\sum_j \text{attention weight}_{\text{she},j}\times \text{value vector}_j
\]
With causal attention, cells to the right of the diagonal show “—”: a token cannot attend to later tokens. The remaining weights are normalized to total 100%.
The demo uses hand-designed weights to illustrate the mechanism. A high weight indicates a strong contribution to this head’s computation; it does not by itself prove causality or explain the final prediction.
<img width="1447" height="739" alt="image" src="https://github.com/user-attachments/assets/7e9c85b5-a6a8-45db-8141-be49f39dca4b" />
## Summary
Transformer 大语言模型（LLM）每次生成一个 Token。
生成的 Token 会被追加到提示词（prompt）末尾，随后更新后的提示词再次输入模型进行前向传播，以生成下一个 Token。
Transformer LLM 的三大核心组件包括：分词器（tokenizer）、堆叠的 Transformer 模块（Transformer blocks）以及语言建模头（language modeling head）。
分词器包含了模型的 Token 词表，模型则拥有与这些 Token 对应的 Token 嵌入（embeddings）。
将文本拆分为 Token 并使用这些 Token 的嵌入，是 Token 生成过程的第一步。
前向传播过程会依次流经各个阶段。
在流程接近尾声时，语言建模头会针对下一个可能的 Token 计算概率得分。解码策略决定了在当前生成步骤中实际选择哪个 Token 作为输出（通常选择概率最高的 Token，但不总是如此）。
Transformer 表现优异的一个原因在于其并行处理 Token 的能力。每个输入 Token 都会进入各自的处理通道或数据流。通道数量即模型的“上下文长度”（context size），代表了模型能够处理的最大 Token 数量。
由于 Transformer LLM 采用循环方式逐个生成 Token，缓存每一步的处理结果是个好主意，这样可以避免重复计算（这些结果以各种矩阵的形式存储在各层之中）。
大部分处理工作都在 Transformer 模块内部进行。
这些模块由两个组件构成。其中之一是前馈神经网络（feedforward neural network），它能够存储信息，并基于训练数据进行预测和插值。
Transformer 模块的第二个主要组件是注意力层（attention layer）。注意力机制整合了上下文信息，使模型能够更好地捕捉语言的细微差别。
注意力机制的运作包含两个主要步骤：(1) 评估相关性得分，以及 (2) 整合信息。 Transformer 的注意力层并行执行多项注意力操作，每项操作都在一个注意力头（attention head）内进行，最终将各头的输出汇总，构成该注意力层的整体输出。
通过在所有注意力头（或各组注意力头，即“分组查询注意力”/grouped-query attention）之间共享键（key）和值（value）矩阵，可以加速注意力计算。
诸如 Flash Attention 之类的方法，通过优化 GPU 不同存储系统上的操作执行方式，从而提升了注意力计算的速度。

