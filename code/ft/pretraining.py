# -*- coding: utf-8 -*-
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset, Trainer, TrainingArguments, RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
import os
from tokenizers import ByteLevelBPETokenizer
import tqdm
from tokenizers.processors import BertProcessing
import argparse

def train_tokenizer(data_p, save_p):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.enable_padding(direction='right',length=128)
    tokenizer.enable_truncation(max_length=128)
    tokenizer.train(files=data_p, vocab_size=10000, min_frequency=2,special_tokens=[
      "<s>",
      "<pad>",
      "</s>",
      "<unk>",
      "<mask>",
    ])
    tokenizer._tokenizer.post_processor = BertProcessing(
      ("</s>", tokenizer.token_to_id("</s>")),
      ("<s>", tokenizer.token_to_id("<s>")),
    )
    pr_path =os.path.join(save_path, 'tokenizer')
    if os.path.exists(pr_path):
        path = pr_path
    else:
        path = os.mkdir(pr_path)

    tokenizer.save_model('{}/'.format(path))

def train_model(data_p, save_p, tokenizer_p):
    config = RobertaConfig(
        vocab_size=10000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizer.from_pretrained("{}".format(tokenizer_p), max_len=512)
    model = RobertaForMaskedLM(config=config)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_p,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )

    training_args = TrainingArguments(
        output_dir="{}".format(save_path),
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=32,
        save_steps=10000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model('pretrained_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, required=False)
    parser.add_argument('--data_path', type=str, required=True)      
    parser.add_argument('--save_path', type=str, required=True)
    args= parser.parse_args()
    data=args.data_path
    save_path=args.save_path
    tokenizer_path = args.tokenizer_path
    train_tokenizer(data, save_path)
    train_model(data, save_path, tokenizer_path)
