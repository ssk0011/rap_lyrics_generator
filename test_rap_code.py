import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import re

artist_list = [
    'JAY-Z',
    'Eminem',
    'Kendrick Lamar',
    'Lil Wayne',
    'Nicki Minaj',
    'Snoop Dogg',
    'Nas',
    'Drake'
]

artist_dict = {}

# for index, row in df.iterrows():
for artist in artist_list:
    artist_filter = df[df['artist'].isin([artist])]['lyrics'].tolist()
    artist_filter_corpus = '\n'.join([str(item) for item in artist_filter])
    # print(artist_filter_corpus)
    # artist_filter_text = artist_filter
    artist_filter_text = re.sub('\[(.*).\]', '', artist_filter_corpus)
    artist_filter_text = artist_filter_text.replace('\n\n', '\n')
    artist_filter_text = artist_filter_text.replace('\n\n', '\n')
    artist_filter_text = artist_filter_text.replace('\n\n', '\n')
    # print(artist_filter_text)
    file_path = "./" + artist.lower() + "_" + "full_corpus.txt"
    with open(file_path, "w") as text_file:
        text_file.write(artist_filter_text)
    # artist_dict[artist] = file_path

    # Initiate the GPT-2 pre-trained model, plus the tokenizer
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Tokenize the lyrics and prepare dataset
    
    # We'll set up the dataset through the tokenizer, referring to
    # the file we just wrote as the basis.
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,  # Save the all_lyrics string to a file and provide its path here
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments; these can be modified depending on
    # available architecture.
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Initiate the Trainer function and start training!
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()

    artist_dict[artist] = model

    input_text = "In a cosmic sort of way"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = artist_dict['JAY-Z'].generate(input_ids, max_length=100, num_return_sequences=5, temperature=0.9, do_sample=True)
    
    for i, text in enumerate(output):
        print(f"Generated Text {i+1}: {tokenizer.decode(text)}")
        print()
    
    break