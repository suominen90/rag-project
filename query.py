import os
import logging
from mistralai import Mistral
from get_vector_database import get_vector_db

api_key = os.getenv('MISTRAL_API_KEY')

logger = logging.getLogger(__name__)

def run_mistral(client, user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)


# Basic querying the vector db and calling LLM API to generate answer
def query(input):
    if input:
        # Initiate mistral
        client = Mistral(api_key=api_key)

        # Get the vector database instance
        db = get_vector_db()

        # Search database
        results = db.similarity_search_with_score(input, k=5)
        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        PROMPT = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {input}
        Answer:
        """

        response = run_mistral(client, PROMPT)
        return response

    return None
