from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# dummy data setup
class OfflineRLDataset(Dataset):
    def __init__(self):
        # each item is tuple: (prompt, description, reward)
        self.data = [
            (
                "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"\u00e2\u2013\u00a2\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\n\nDistractor grid 1:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 X\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\n\nDistractor grid 2:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
                "Expression: F",
                "Answer: 3rd",
                1
            ),
            (
                "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"\u00e2\u2013\u00a2\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 X\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\n\nDistractor grid 1:\n\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
                "Expression: P",
                "Answer. 3.",
                -1
            ),
            (
                "You are given three grids, where each of them is 5 by 5 in size.\nGrids have empty cells marked with \"\u00e2\u2013\u00a2\" and filled cells marked with \"X\".\nYour task is to generate a referring expression that best describes the target grid while distinguishing it from the two other distractor grids.\nThe first grid is the target grid, and the following two grids are the distractors.\n\nTarget grid:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\n\nDistractor grid 1:\n\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\nX X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X X X\n\nDistractor grid 2:\n\nX X X X X\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX X X \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\nX \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1 \u00e2\u2013\u00a1\n\nInstruction: Describe the target grid.\nGenerate the referring expression starting with the tag \"Expression: \" for the given target grid. Omit any other text.",
                "Expression: C",
                "Answer: Thirrd",
                -1
            )
        ]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# Load a small model for testing
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.train()

dataset = OfflineRLDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = Adam(model.parameters(), lr=5e-5)


## Offline REINFORCE Loop
for epoch in range(3):  # small number of epochs for testing
    for prompt, description, target, reward in dataloader:
        # Tokenize prompt + description as a single sequence
        inputs = tokenizer(prompt[0], return_tensors="pt")
        labels = tokenizer(description[0], return_tensors="pt")["input_ids"]

        # Concatenate prompt and description
        input_ids = torch.cat([inputs["input_ids"], labels], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones_like(labels)], dim=1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        log_probs = -F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, model.config.vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction='none'
        ).view(input_ids.shape[0], -1)  # Shape: [batch, seq_len]

        # Get log-prob of only the "description" tokens (not prompt)
        prompt_len = inputs["input_ids"].shape[1]
        description_log_probs = log_probs[:, prompt_len-1:].sum(dim=1)

        # Multiply by reward (REINFORCE objective)
        loss = -reward[0] * description_log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
