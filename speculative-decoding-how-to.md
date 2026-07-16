# How to Implement Speculative Decoding

Speculative decoding is an inference-time optimization that reduces latency by letting a **small draft model** guess several upcoming tokens and then asking the **large target model** to verify those guesses in a single pass.

The practical idea is simple:

- Standard decoding: **1 expensive forward pass -> 1 token**
- Speculative decoding: **1 expensive forward pass -> several accepted tokens**

The big model still controls the final output. The small model only proposes candidates.

---

## 1. What You Need

A basic speculative decoding implementation needs three things:

### Models
- **Target model**: the main large language model whose quality you want to preserve.
- **Draft model**: a much smaller and faster model that can propose likely next tokens.

### Shared tokenizer
Both models must use the **same tokenizer and vocabulary**. If tokenization differs, the draft tokens cannot be validated correctly by the target model.

### Runtime loop
You need a decoding loop that can:
- generate draft tokens,
- run one verification pass on the target model,
- accept matching tokens,
- and fall back to normal decoding when the draft diverges.

---

## 2. High-Level Algorithm

At each generation step:

1. Start from the current sequence.
2. Ask the **draft model** to generate the next `K` tokens.
3. Feed the full drafted continuation into the **target model**.
4. Compare each drafted token against the target model's distribution.
5. Accept the longest valid prefix.
6. If a token is rejected, let the target model generate the next token normally.
7. Repeat until the sequence ends.

This means the system often advances multiple tokens after a single expensive target-model pass.

---

## 3. Step-by-Step Walkthrough

### Step 1: Generate draft tokens
Suppose the current context is:

```text
Prompt: "Explain speculative decoding in simple terms"
```

The draft model proposes the next `K=4` tokens:

```text
["as", "a", "faster", "method"]
```

These are only guesses.

### Step 2: Verify with the target model
Now send the original context plus the drafted continuation to the target model in one forward pass.

The target model computes the next-token distributions for each drafted position:

- `P_target("as" | context)`
- `P_target("a" | context + "as")`
- `P_target("faster" | context + "as a")`
- `P_target("method" | context + "as a faster")`

### Step 3: Accept the valid prefix
If the target model agrees with the first three tokens but not the fourth:

- Accept: `"as"`, `"a"`, `"faster"`
- Reject: `"method"`

Then let the target model produce the correct token at the rejection point.

### Step 4: Resume from the corrected sequence
Continue drafting again from the new sequence.

---

## 4. Pseudocode

```python
final_tokens = tokenize(prompt)

while not done:
    # Draft phase
    draft_tokens = []
    ctx = final_tokens.copy()
    for _ in range(K):
        next_token = draft_model.sample_next(ctx)
        draft_tokens.append(next_token)
        ctx.append(next_token)

    # Verification phase
    logits = target_model.forward(final_tokens + draft_tokens)

    accepted_all = True
    for i, d_token in enumerate(draft_tokens):
        probs = softmax(logits[i])
        target_token = sample_from_target_distribution(probs)

        if target_token == d_token:
            final_tokens.append(d_token)
        else:
            final_tokens.append(target_token)
            accepted_all = False
            break

    if accepted_all:
        continue
```

This is simplified pseudocode, but it captures the core loop.

---

## 5. Acceptance Logic

The acceptance rule is the critical part.

A correct speculative decoder must preserve the **target model's output distribution**. That means the draft model cannot simply override the target model whenever it looks plausible.

In practice:
- The target model remains the source of truth.
- The draft token is accepted only when the verification rule says it is consistent.
- If the draft diverges, the target model corrects the sequence immediately.

This is what makes speculative decoding different from simply replacing the large model with a smaller one.

---

## 6. Why It Works

Large-model generation is often bottlenecked by repeated autoregressive passes.

Speculative decoding improves performance by shifting some of the work to a cheaper model and then verifying multiple candidate tokens at once.

The speedup comes from two conditions:

1. The **draft model is significantly faster** than the target model.
2. The **acceptance rate is high enough** that several drafted tokens survive verification.

If both conditions hold, the target model performs fewer expensive passes per generated token.

---

## 7. Where It Helps Most

Speculative decoding is most useful when:

- outputs are long,
- latency matters,
- the target model is large,
- and the hardware is not already saturated with very large batch sizes.

Typical good fits:
- chat systems,
- coding assistants,
- agent loops,
- document generation,
- high-traffic inference APIs.

---

## 8. Common Failure Modes

### Draft model is too large
If the draft model is too slow, the extra drafting overhead can cancel out the speedup.

### Draft model is too weak
If it guesses poorly, too many tokens get rejected and the target model ends up doing nearly the same amount of work as standard decoding.

### Tokenizer mismatch
If draft and target models do not share the same tokenizer, speculative verification breaks.

### Wrong deployment assumptions
Speculative decoding is not guaranteed to help every workload. It should be benchmarked under realistic batch size, concurrency, and prompt-length conditions.

---

## 9. Practical Implementation Options

Instead of writing the full runtime from scratch, most teams use inference frameworks that already support speculative decoding, such as:

- **vLLM**
- **TensorRT-LLM**
- **Triton Inference Server**
- other optimized serving stacks

This is usually the fastest path to production.

If building a custom implementation, start with:
- a large target model,
- a smaller draft model from the same family when possible,
- shared tokenizer validation,
- and a benchmark harness to measure latency, throughput, and acceptance rate.

---

## 10. Minimal Mental Model

A simple way to remember speculative decoding:

> The small model guesses ahead.
> The large model checks those guesses.
> Everything the large model agrees with is accepted.
> The large model still decides the final answer.

That is why speculative decoding can improve latency **without changing the behavior of the target model**.
