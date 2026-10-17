# ICT4403 Assignment 2: Autoregressive Text Generation

Compare word-level next-token Simple RNN, Bidirectional RNN, LSTM, and GRU models on Tiny Shakespeare. Dataset: [Karpathy Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).

## Method
Each model uses a 24-word context, 96-dimensional embeddings, 128 hidden units, batch size 64, Adam (0.001), and 8 epochs × 500 updates. The text is tokenized into words and punctuation, lowercased, and split sequentially into 80% train, 10% validation, 10% test. Vocabulary is built from training tokens only (5,957 entries including <unk>). Final metrics are held-out test cross-entropy, perplexity, next-token accuracy, and measured runtime.

## Results
| Model | Test loss | Perplexity | Accuracy | Time (s) |
|---|---:|---:|---:|---:|
| Simple RNN | 4.923 | 137.45 | 16.62% | 23.17 |
| Bidirectional RNN | 4.953 | 141.56 | 14.82% | 39.73 |
| LSTM | 4.846 | 127.29 | 14.63% | 45.95 |
| GRU | 4.791 | 120.42 | 17.33% | 38.45 |

The GRU led on test loss, perplexity, and accuracy; the Simple RNN trained fastest. Samples from all models remain fragmentary. Temperature samples at 0.6, 0.9, and 1.2 show that lower temperature tends to repeat common words and fragments; 1.2 gives more varied but less coherent text. See results/generated_samples.md and plots in results/.

## Run
Python 3.10+: create a virtual environment, install requirements.txt, then run python train.py. The default run uses 8 epochs × 500 steps. The --quick option is only a pipeline check.

The bidirectional RNN summarizes only the completed context before predicting the next token, so it does not see the target. Results were generated with seed 4403 on CPU using Tiny Shakespeare (1,115,394 characters; 253,593 tokens). See train.py, requirements.txt, and results/ for details.
