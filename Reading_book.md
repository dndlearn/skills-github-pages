## One chart is better than hundreds words

-- Understand the branch
Key Concepts:

main Branch: Usually the trusted working version, and the first branch. (historically called master)
Feature Branch: A safe isolated space to develop without affecting the trusted version.
Merging: Combining changes from different branches.
![Branch Diagram](https://x.com/DeRonin_/article/2033587293064204349)


after the merge
![Branch Diagram](https://x.com/DeRonin_/article/2033587293064204349)

LLM concept:


## How the Vector Values standardly work:
-1 (Present): A value of 1 means that word exists in the input sentence.Words present: is, cute, my, cat $\rightarrow$ mapped to [1, 1, 1, 1].   
-0 (Absent): A value of 0 means that word does not appear in the input sentence.Words absent: that, a, dog $\rightarrow$ mapped to [0, 0, 0].   
<img width="1341" height="737" alt="image" src="https://github.com/user-attachments/assets/b7a6a5e7-f797-4c2b-abc5-13ebda0aaa6a" />
