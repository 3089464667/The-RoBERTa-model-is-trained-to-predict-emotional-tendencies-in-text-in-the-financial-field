import praw
import time
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch



# 认证并连接到Reddit API
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# 加载RoBERTa情感分析模型和tokenizer
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def fetch_reddit_posts(subreddit_name, query, limit=10):
    posts_data = []
    subreddit = reddit.subreddit(subreddit_name)
    
    for submission in subreddit.search(query, limit=limit):
        text = submission.title + " " + submission.selftext
        # 使用RoBERTa模型进行情感分析
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # 根据预测结果分配标签
        label = predicted_class_id
        
        posts_data.append({
            'text': text,
            'label': label
        })
        time.sleep(1)
    
    return posts_data

# 公司列表
companies = ["Sands China Ltd", "Galaxy Entertainment Group Limited", 
             "MGM China Holdings Limited", "Wynn Macau Limited", 
             "Melco International Development Limited", "SJM Holdings Limited"]

subreddits = ["investing", "stocks", "wallstreetbets", "financialindependence", "cryptocurrency", "stockmarket", "personalfinance"]
all_posts = []

# 遍历每个subreddit和每个公司
for subreddit_name in subreddits:
    for company in companies:
        print(f"Fetching posts from /r/{subreddit_name} with keyword '{company}'")
        posts = fetch_reddit_posts(subreddit_name, company, limit=10)
        all_posts.extend(posts)

# 转换为DataFrame
df = pd.DataFrame(all_posts)

# 转换为Hugging Face Datasets格式
dataset = Dataset.from_pandas(df)

# 数据预处理
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='results_4',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset.train_test_split(test_size=0.1)['train'],
    eval_dataset=encoded_dataset.train_test_split(test_size=0.1)['test'],
)

# 训练模型
trainer.train()

# 评估模型
trainer.evaluate()

# for post in all_posts:
#     print(f"Text: {post['text']}\nPredicted Label: {post['label']}\n")


# 保存更新后的模型
model.save_pretrained('updated4_sentiment_model')
tokenizer.save_pretrained('updated4_sentiment_model')

'''
Label 0: 表示负面情感。
Label 1: 表示中性情感。
Label 2: 表示正面情感。
'''