Reproduce Emotion Vector Paper
====

TL;DR: The repository is to reproduce the [emotion vector paper](https://transformer-circuits.pub/2026/emotions/index.html) by Anthropic with Llama 3.1 8B Instruct. The reproduction is not just repeating, but also understanding the generalization of the method.

Part 1 Report: [PDF](https://jyhong.gitlab.io/post/emotion-vector-part1/report/part1_report.pdf), [blog](https://jyhong.gitlab.io/post/emotion-vector-part1/)

## Motivation

I am a fan of auto-research, not just auto-search for hyperparameters, but the bigger research loop from idea to publication. 
The very first step of many research is to reproduce a high-quality paper.
So when I read Anthropic's paper, I was excited to reproduce it with Llama 3.1 8B Instruct, an open-weight model that I can afford.

Why auto-reproducing experiments is important for me? 
* *No data contamination*: Proof of concept on a never-trained data (paper).
* *Faster to reproduce* high-tech study in my toy models and explore my ideas.
* *Faster to know* if an idea in giant model can work on smaller ones.

What I found works well are:
* Coding wise is good. I don't need to spend a lot of time in reading details and check the parameters.

What I found still hard/critical:
* Review is still a lot of efforts. Even Claude is bad at understanding the visual difference, really bad at aligning the figures.
* Define clear success criteria, otherwise the agent is not able to know when (not) to stop.
* Ask a good question. Reproducing a paper on a different model is not simply repeating a experiment. It is about understanding why the difference exists. For example, is the difference from the code or the model?



