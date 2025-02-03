import os
import openai
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your environment variable is set

# Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the Pinecone index name
index_name = "sample"

# Check if the index exists, create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on OpenAI's text embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Handle possible None values
    return text

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response['data'][0]['embedding']

# Directory containing PDF files
pdf_directory = "com/data"  # Update with the correct path

# Process each PDF in the directory
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)

        print(f"Extracting text from {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)

        print(f"Generating embeddings for {pdf_file}...")
        embedding = generate_embedding(text)

        print(f"Pushing data from {pdf_file} into Pinecone...")
        index.upsert([
            (pdf_file, embedding, {"source": "Local PDF", "text": text[:500]})
        ])

print("All PDFs processed and data pushed to Pinecone successfully!")

# Query the Pinecone index
query_text = "Find documents similar to this."
query_embedding = generate_embedding(query_text)
query_results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)

print("Query Results:")
for match in query_results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")
