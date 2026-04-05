import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

class RAGEngine:
    def __init__(self, model_name="google/flan-t5-base", embedding_model="all-MiniLM-L6-v2"):
        print(f"Initializing RAG Engine with {model_name}...")
        self.encoder = SentenceTransformer(embedding_model)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.llm_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="production_history")

    def row_to_document(self, row):
        verif_status = "has verification data" if row.get('Has_Verification', False) else "has NO verification data"
        defect_info = f"Defect Code: {row['DefectCode']}" if pd.notna(row.get('DefectCode')) and row.get('DefectCode') not in ['', 'None'] else "No defects recorded"
        return (
            f"Board Barcode: {row['Barcode']}. "
            f"Pattern: {row['Pattern_Barcode']}. "
            f"Component: {row['Designator']} placed by Feeder {row['Feede_ID']} using Nozzle {row['Nozel_Name']}. "
            f"Result: {row['Componenet_Result']}. "
            f"This board {verif_status}. "
            f"{defect_info}. "
            f"Produced on {row['NPM_Date']}."
        )

    def index_data(self, df, sample_size=10000):
        print(f"Indexing {sample_size} records into ChromaDB...")
        df_sample = df.dropna(subset=['Barcode', 'Designator']).head(sample_size)
        documents = [self.row_to_document(row) for _, row in df_sample.iterrows()]
        embeddings = self.encoder.encode(documents, show_progress_bar=True).tolist()
        ids = [str(i) for i in range(len(documents))]
        
        # Batch addition
        BATCH_SIZE = 5000
        for i in range(0, len(documents), BATCH_SIZE):
            self.collection.add(
                documents=documents[i:i+BATCH_SIZE],
                embeddings=embeddings[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE]
            )
        print("✅ Indexing complete!")

    def query(self, user_question, top_k=5):
        query_embedding = self.encoder.encode([user_question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        retrieved_docs = results['documents'][0]
        
        context = "\n".join(retrieved_docs)
        prompt = f"Based on this manufacturing data:\n{context}\n\nAnswer this question: {user_question}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.llm_model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response, retrieved_docs
