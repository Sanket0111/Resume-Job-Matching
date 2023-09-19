#!/usr/bin/env python
# coding: utf-8

# # Step 1: Import necessary libraries.

# In[1]:


# Import os library to print the full path of all the files in the directory `E:/Projects/`
import os
for dirname, _, filenames in os.walk('E:/Projects/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd                      ##These lines import additional libraries that will be used in the code:    
import numpy as np
import sklearn as sns                    #Scikit-learn for machine learning
import matplotlib.pyplot as plt          #Matplotlib for plotting
from tqdm.auto import tqdm               #TQDM for progress bars
import datasets                          #Datasets and Transformers for natural language processing
import transformers


# In[3]:


pip install numpy --upgrade


# # Step 2: Load resume and job description data.

# In[4]:


# Read the resume data from a CSV file and drop the `Resume_html` column
resume_data=pd.read_csv(r"E:\Projects\Resume.csv")
resume_data=resume_data.drop(["Resume_html"],axis=1)


# In[5]:


#This function extracts the text from a PDF file.
from pypdf import PdfReader

def pdf_text(filePath:str)->str:
    reader = PdfReader(filePath)
    text=""
    for page in reader.pages:
        text+=page.extract_text()
    return text


# # Step 3: Preprocess the text of the resumes and job descriptions.

# In[6]:


# Define a function to preprocess the text
from nltk import pos_tag
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import string
import re

puncuation=set(string.punctuation)
stop_words_english=set(stopwords.words("english"))
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}

    for sent in sentences:
        for criteria in ['skills', 'education']:
            if criteria in sent:
                words = word_tokenize(sent)
                words = [word for word in words if word not in stop_words_english]
                # Use a part-of-speech (POS) tagger to identify and remove stop words and other irrelevant words.
                tagged_words = pos_tag(words)
                filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
                features['feature'] += " ".join(filtered_words)

    return features


# In[7]:


progress_bar=tqdm(range(len(resume_data)))
# Extract text from the CSV or PDF file.
# If the `Resume_str` column is present in the CSV file, use that.
# Otherwise, use the `pdf_text` function to extract text from the PDF file.
def process(df):    
    id=df['ID']
    category=df['Category'] 
    text=pdf_text(f"E:/Projects/data/data/{category}/{id}.pdf")   
    features=preprocess_text(text)
    df['Feature']=features['feature']
    progress_bar.update(1)
    return df
# Process the resume data.
resume_data=resume_data.apply(process,axis=1)
resume_data=resume_data.drop(columns=['Resume_str'])


# In[27]:


# Use the NLTK averaged perceptron tagger to identify and remove stop words and other irrelevant words.
import nltk
nltk.download('averaged_perceptron_tagger')


# In[9]:


# Save the processed resume data to a CSV file to avoid recomputing it in the future.
resume_data.to_csv(r"E:\Projects\Resume.csv",index=False)


# In[10]:


resume_data=pd.read_csv(r"E:\Projects\Resume.csv")


# In[11]:


resume_data.head()


# In[12]:


resume_data['Category'].value_counts().sort_index().plot(kind="bar",figsize=(12,6))


# In[13]:


# Fetch the job descriptions from the job description data.
num_desc=15 # Number of job descriptions to fetch.

# Read the job description data from the CSV file.
job_description=pd.read_csv(r"E:\Projects\training_data.csv")

# Select 15 `num_desc` job descriptions.
job_description=job_description[["job_description","position_title"]][:num_desc]


# In[14]:


# Preprocess the text of the job descriptions.
# This involves converting the text to lowercase, removing punctuation, and stop words, and lemmatizing the words.
job_description['Features']=job_description['job_description'].apply(lambda x : preprocess_text(x)['feature'])


# In[15]:


# Select the first/any 15 job descriptions and preprocess the text.
# Prepare the job descriptions for training.
job_description=job_description[["job_description","position_title"]][:num_desc]
job_description['Features']=job_description['job_description'].apply(lambda x : preprocess_text(x)['feature'])


# In[16]:


job_description.head()


# # Step 4: Generate embeddings for the resumes and job descriptions.

# In[17]:



from transformers import AutoModel, AutoTokenizer
import torch
# Load the BERT base uncased model and tokenizer.
device="cuda"if torch.cuda.is_available() else "cpu"

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move the model and tokenizer to the GPU for faster processing. 
model.to(device)


def get_embeddings(text):
    inputs = tokenizer(str(text), return_tensors="pt",truncation=True,padding=True).to(device)  # Tokenize the text and convert it to a PyTorch tensor.
    outputs = model(**inputs)    # Get the model's output.
     # Get the mean of the last hidden state of the model
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().to("cpu").numpy() 
    return embeddings


# In[19]:


# Calculate the cosine similarity between the job description embeddings and the resume embeddings.

# Import the cosine similarity function from scikit-learn.
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# Calculate embeddings for all job descriptions and resumes
job_desc_embeddings = np.array([get_embeddings(desc) for desc in job_description['Features']])
resume_embeddings = np.array([get_embeddings(text) for text in resume_data['Feature']])


# In[20]:


# Squeeze the job description and resume embeddings to remove the extra dimension.
job_desc_embeddings=job_desc_embeddings.squeeze()
resume_embeddings=resume_embeddings.squeeze()
# Print the shape of the job description and resume embeddings.
resume_embeddings.shape,job_desc_embeddings.shape


# In[21]:


# Initialize a Pandas DataFrame to store the results of the job recommendation system.
# The DataFrame has five columns:
# * jobId: The ID of the job.
# * resumeId: The ID of the resume.
# * similarity: The cosine similarity between the job description and the resume.
# * domainResume: The domain of the resume.
# * domainDesc: The domain of the job description.
result_df = pd.DataFrame(columns=['jobId', 'resumeId', 'similarity', 'domainResume', 'domainDesc'])
# Set the number of top resumes to recommend for each job.
k=5


# In[22]:


# Iterate over the job descriptions and compute the cosine similarity between each job description and all resumes.
# For each job description, extract the top-k most similar resumes and add the relevant information to the result DataFrame.
for i, job_desc_emb in enumerate(job_desc_embeddings):
    job_desc_id = i
    job_title = job_description['position_title'].iloc[i]
     
    # Compute cosine similarities between the current job description and all resumes.
    similarities = cosine_similarity([job_desc_emb], resume_embeddings )

    # Get the indices of the top-k most similar resumes.
    top_k_indices = np.argsort(similarities[0])[::-1][:k]
   
    # Extract the relevant information and add it to the result DataFrame.
    for j in top_k_indices:
        resume_id = resume_data['ID'].iloc[j]
        work_domain = resume_data['Category'].iloc[j]
        similarity_score = similarities[0][j]
        
        result_df.loc[i+j] = [job_desc_id, resume_id, similarity_score, work_domain,job_title ]
        

# Sort the results by similarity score (descending).
result_df = result_df.sort_values(by='similarity', ascending=False)


# In[23]:


result_df.head()


# In[24]:


result_group=result_df.groupby("jobId")
result_group


# In[25]:


# Get the first group of results.
result_group.get_group(0)


# In[26]:


# Print the top 15 resumes for each job description.

for i in range(num_desc):
    print()
    print("jobId---cosineSimilarity---domainResume---domainDesc")
    print(result_group.get_group(i).values[0])
    print()


# # -------------------------------FINISHED----------------------------------------

# In[ ]:




