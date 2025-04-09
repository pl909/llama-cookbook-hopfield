# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/narrative_qa

import copy
import random
import datasets

from unittest.mock import patch

@patch('builtins.input', return_value="N")
def load_narrative_qa(split, _):
    try:
        ds = datasets.load_dataset("narrative_qa", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
          raise ValueError("Loading narrative_qa requires you to execute the dataset script in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
          raise e
    return ds



def select_answer(answers, strategy="random"):
    """
    Select an answer from a list of answers using the specified strategy.
    
    Args:
        answers: List of answer objects, each with a "text" field
        strategy: One of ["random", "first", "longest", "shortest", "consensus"]
    
    Returns:
        Selected answer text string
    """
    if not answers:
        return ""
    
    # Extract answer texts
    answer_texts = [answer["text"] for answer in answers]
    
    if strategy == "random":
        return random.choice(answer_texts)
    elif strategy == "first":
        return answer_texts[0]
    elif strategy == "longest":
        return max(answer_texts, key=len)
    elif strategy == "shortest":
        return min(answer_texts, key=len)
    elif strategy == "consensus":
        # Simple consensus approach: find the answer with the most word overlap with others
        if len(answer_texts) == 1:
            return answer_texts[0]
        
        # Convert answers to sets of words
        answer_word_sets = [set(answer.lower().split()) for answer in answer_texts]
        
        # Calculate overlap scores
        overlap_scores = []
        for i, word_set in enumerate(answer_word_sets):
            # Calculate how many words this answer shares with other answers
            overlap = sum(len(word_set.intersection(other_set)) for j, other_set in enumerate(answer_word_sets) if i != j)
            overlap_scores.append((overlap, i))
        
        # Return the answer with the highest overlap
        _, best_idx = max(overlap_scores)
        return answer_texts[best_idx]
    else:
        # Default to first answer
        return answer_texts[0]


def get_memory_narrative_qa(dataset_config, tokenizer, split, answer_strategy="random"):
    """
    Format the Narrative QA dataset for memory-based training where the story and question
    are sent separately to test the model's memory performance.
    
    Args:
        dataset_config: Configuration for the dataset
        tokenizer: Tokenizer to use
        split: Dataset split to use
        answer_strategy: Strategy for selecting answers from multiple options
    """
    dataset = load_narrative_qa(split)

    # Create separate prompts for story and question
    story_prompt = (
        f"{{document}}\n\n" # maybe add supporting formatting
    )
    
    question_prompt = (
        f"Please answer the following user's question:\n\n"
        f"Question: {{question}}\n\n"
        f"Answer: "
    )

    def apply_memory_prompt_template(sample):
        return {
            "story_prompt": story_prompt.format(
                document=sample["document"]["text"]
            ),
            "question_prompt": question_prompt.format(
                question=sample["question"]["text"]
            ),
            "answer": select_answer(sample["answers"], strategy=answer_strategy),
        }

    dataset = dataset.map(apply_memory_prompt_template, remove_columns=list(dataset.features))

    def tokenize_memory_add_label(sample):
        # Tokenize the story prompt
        story_tokens = tokenizer.encode(tokenizer.bos_token + sample["story_prompt"], add_special_tokens=False)
        
        # Tokenize the question prompt and answer
        question_tokens = tokenizer.encode(sample["question_prompt"], add_special_tokens=False)
        answer_tokens = tokenizer.encode(sample["answer"] + tokenizer.eos_token, add_special_tokens=False)
        
        # Combine question and answer tokens
        qa_tokens = question_tokens + answer_tokens
        
        # Create the full input sequence
        input_ids = story_tokens + qa_tokens
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Create labels: -100 for story and question, actual tokens for answer
        labels = [-100] * len(story_tokens) + [-100] * len(question_tokens) + answer_tokens

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # Add separate fields for story and question for potential use in training
            "story_length": len(story_tokens),
            "question_length": len(question_tokens),
        }

        return sample

    dataset = dataset.map(tokenize_memory_add_label, remove_columns=list(dataset.features), batched=True, cache_file_names=f"narrative_qa_{split}_memory_dataset.cache")

    return dataset
