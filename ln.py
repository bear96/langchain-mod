from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
import logging
logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta
from typing import List
from termcolor import colored
import math
import faiss

#pydantic, tenacity, openapi_schema_pydantic, sqlalchemy, numexpr, mypy_extensions

USER_NAME = "Person A" # The name you want to use when interviewing the agent.
# openai.openai_api_key = "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"
LLM = ChatOpenAI(openai_api_key = "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",
                 temperature=0.5, max_tokens=1500) # Can be any LLM you want.



def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings(openai_api_key="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")
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)    

tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8 # we will give this a relatively low number to show how reflection works
)

tommie = GenerativeAgent(name="Tommie", 
              age=25,
              traits="anxious, likes design, talkative", # You can add more persistent traits here 
              status="looking for a job", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              memory=tommies_memory
             )

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
# print(tommie.get_summary())

# We can add memories directly to the memory object
tommie_observations = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)

# We will see how this summary updates after more observations to create a more rich description.
print(tommie.get_summary(force_refresh=True,now=datetime.now()))