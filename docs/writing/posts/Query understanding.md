---
date: 2024-08-01
readtime: 11
authors:
- Dewey
categories:
- RAG
---

# Query understanding

Query understanding is a complicated issue. It involves not only processing the query itself but also the procedure for constructing data indexes and retrieval.

So, I will follow these steps:

- Constructing data indexes
- Retrieval
- Query understanding

## Constructing data indexes

One major limitation is that the query we embed and the text we search for often don't have a direct match, leading to suboptimal results. A common method to improve this is to extract information from a document and use it to answer a question. We can also be creative in how we extract, summarize, and generate potential questions to improve our embeddings.

For example, instead of using just a text chunk we could try to:

1. extract key words and themes
2. extract hypothetical questions
3. generate a summary of the text

```python
class Extraction(BaseModel):
    topic: str
    summary: str
    hypothetical_questions: List[str] = Field(
        default_factory=list,
        description="Hypothetical questions that this document could answer",
    )
    keywords: List[str] = Field(
        default_factory=list, description="Keywords that this document is about"
    )

text_chunk = """
## Simple RAG

**What is it?**

The simplest implementation of RAG embeds a user query and do a single embedding search in a vector database, like a vector store of Wikipedia articles. However, this approach often falls short when dealing with complex queries and diverse data sources.

**What are the limitations?**

- **Query-Document Mismatch:** It assumes that the query and document embeddings will align in the vector space, which is often not the case.
    - Query: "Tell me about climate change effects on marine life."
    - Issue: The model might retrieve documents related to general climate change or marine life, missing the specific intersection of both topics.
- **Monolithic Search Backend:** It relies on a single search method and backend, reducing flexibility and the ability to handle multiple data sources.
    - Query: "Latest research in quantum computing."
    - Issue: The model might only search in a general science database, missing out on specialized quantum computing resources.
- **Text Search Limitations:** The model is restricted to simple text queries without the nuances of advanced search features.
    - Query: "what problems did we fix last week"
    - Issue: cannot be answered by a simple text search since documents that contain problem, last week are going to be present at every week.
- **Limited Planning Ability:** It fails to consider additional contextual information that could refine the search results.
    - Query: "Tips for first-time Europe travelers."
    - Issue: The model might provide general travel advice, ignoring the specific context of first-time travelers or European destinations.
"""

extractions = client.chat.completions.create(
    model="gpt-4-1106-preview",
    stream=True,
    response_model=Iterable[Extraction],
    messages=[
        {
            "role": "system",
            "content": "Your role is to extract chunks from the following and create a set of topics.",
        },
        {"role": "user", "content": text_chunk},
    ],
)

for extraction in extractions:
    pprint(extraction.model_dump())
```

```
{'hypothetical_questions': ['What is the basic concept behind simple RAG?',
                            'How does simple RAG work for information '
                            'retrieval?'],
 'keywords': ['Simple RAG',
              'Retrieval-Augmented Generation',
              'user query',
              'embedding search',
              'vector database',
              'Wikipedia articles',
              'information retrieval'],
 'summary': 'The simplest implementation of Retrieval-Augmented Generation '
            '(RAG) involves embedding a user query and conducting a single '
            'embedding search in a vector database, like a vector store of '
            'Wikipedia articles, to retrieve relevant information. This method '
            'may not be ideal for complex queries or varied data sources.',
 'topic': 'Simple RAG'}
{'hypothetical_questions': ['What are the drawbacks of using simple RAG '
                            'systems?',
                            'How does query-document mismatch affect the '
                            'performance of RAG?',
                            'Why is a monolithic search backend a limitation '
                            'for RAG?'],
 'keywords': ['limitations',
              'query-document mismatch',
              'simple RAG',
              'monolithic search backend',
              'text search',
              'planning ability',
              'contextual information'],
 'summary': 'Key limitations of the simple RAG include query-document '
            'mismatch, reliance on a single search backend, constraints of '
            'text search capabilities, and limited planning ability to '
            'leverage contextual information. These issues can result in '
            'suboptimal search outcomes and retrieval of irrelevant or broad '
            'information.',
 'topic': 'Limitations of Simple RAG'}
```

Now you can imagine if you were to embed the summaries, hypothetical questions, and keywords in a vector database (i.e. in the metadata fields of a vector database), you can then use a vector search to find the best matching document for a given query. What you'll find is that the results are much better than if you were to just embed the text chunk!

## Retrieval

Generally, we can use hybrid search (keyword search and embedding search in a single extraction material) together, but we must be careful about how to organize hybrid search. For example, if the HyDE question is not correct, we must reduce the weight of the HyDE question. The organization of hybrid search must be tuned for your specific project.If you are confused about retrieval, you can refer to this link...

But the key to retrieval is providing the proper query. How do you generate the proper query? Let's find out in the next chapter.

## Query understanding

### What is Query Rewriting?

Simply put, query rewriting means we will rewrite the user query in our own words, that our RAG app will know best how to answer. Instead of just doing retrieve-then-read, our app will do a rewrite-retrieve-read approach.

We use a Generative AI model to rewrite the question. This model be a large model, like (or the same as) the one we use to answer the question in the final step. Or it can also be a smaller model, specially trained to perform this task.

Also, query rewriting can take many different forms depending on the needs of the app. Most of the time, basic query rewriting will be enough. But, depending on the complexity of the questions we need to answer, we might need more advanced techniques like HyDE, multi-querying or step-back questions. More information on those in the following section.

### Why does it work?

Query Rewriting usually gives better performance in any RAG app that is knowledge intensive. This is because RAG applications are sensitive to the phrasing and specific keywords of the query. Paraphrasing this query is helpful in the following scenarios:

1. It restructures oddly written questions so they can be better understood by our system.
2. It erases context given by the user which is irrelevant to the query.
3. It can introduce common keywords, which will give it a better chance of matching up with the correct context.
4. It can split complex questions into different sub.questions, which can be more easily responded separately, each with their corresponding context.
5. It can answer question that require multiple levels of thinking by generating a step-back question, which is a higher-level concept question to the one from the user. It then uses both the original and the step-back question to retrieve context.
6. It can use more advanced query rewriting techniques like HyDE to generate hypothetical documents to answer the question. These hypothetical documents will better capture the intent of the question and match up with the embeddings that contain the answer in the vector DB.

### General strategies of query rewriting

Here are some key examples and use cases for various query rewriting techniques in RAG systems:

**Hypothetical Document Embeddings (HyDE)**

HyDE generates hypothetical documents based on the query to improve retrieval[1][2]. The process works as follows:

1. An LLM generates a hypothetical document answering the query
2. This document is encoded into a vector
3. The vector is used for retrieval instead of the original query vector

For example, given the query "What are the effects of climate change?", HyDE might generate a hypothetical document like:

"Climate change has wide-ranging effects on the environment, including rising sea levels, more frequent extreme weather events, and shifts in plant and animal ranges. It impacts agriculture, water resources, human health, and ecosystems globally."

This hypothetical document is then encoded and used for retrieval, potentially improving results compared to using the original short query[1].

**Step-Back Prompting**

Step-back prompting generates broader, more generic queries from a specific query[3]. For example:

Original query: "How do I change a flat tire on a 2015 Toyota Camry?"
Step-back query: "What are the general steps for changing a flat tire on a car?"

This broader query may retrieve more relevant general information about tire changing that can then be applied to the specific case.

**Query2Doc**

Query2Doc generates pseudo-documents to expand and clarify the query[2]. Unlike HyDE, which assumes a direct answer, Query2Doc creates a document that provides context around the query. For example:

Query: "What are the health benefits of turmeric?"
Generated pseudo-document: "Turmeric is a spice commonly used in Indian cuisine. It has been studied for potential health benefits. Some areas of interest include its anti-inflammatory properties, antioxidant effects, and possible impacts on brain function and heart health."

This pseudo-document is then used for retrieval, potentially capturing a wider range of relevant information.

**Multi-querying / Sub-queries**

This technique breaks down complex queries into multiple simpler sub-queries[3]. For example:

Original query: "Compare the economic policies of FDR and Reagan"
Sub-queries:

1. "What were the main economic policies of Franklin D. Roosevelt?"
2. "What were the key economic policies of Ronald Reagan?"
3. "How did FDR's economic approach differ from Reagan's?"

By breaking down the complex query, the system can retrieve more specific and relevant information for each aspect of the question.

These techniques can significantly improve retrieval performance in RAG systems, leading to more accurate and comprehensive answers. The choice of technique depends on the specific use case, query complexity, and the nature of the information being retrieved.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Initialize LLM and embeddings
llm = OpenAI(temperature=0.7)
embeddings = OpenAIEmbeddings()

# 1. Hypothetical Document Embeddings (HyDE)
hyde_prompt = PromptTemplate(
    input_variables=["query"],
    template="Generate a detailed answer for the following question:\n{query}\n\nAnswer:"
)

hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

def hyde(query):
    hypothetical_doc = hyde_chain.run(query)
    doc_embedding = embeddings.embed_query(hypothetical_doc)
    return doc_embedding

# Usage
query = "What are the effects of climate change?"
hyde_embedding = hyde(query)

# 2. Step-Back Prompting
step_back_prompt = PromptTemplate(
    input_variables=["query"],
    template="Generate a more general version of this specific query:\n{query}\n\nGeneral query:"
)

step_back_chain = LLMChain(llm=llm, prompt=step_back_prompt)

def step_back(query):
    general_query = step_back_chain.run(query)
    return general_query

# Usage
specific_query = "How do I change a flat tire on a 2015 Toyota Camry?"
general_query = step_back(specific_query)

# 3. Query2Doc
query2doc_prompt = PromptTemplate(
    input_variables=["query"],
    template="Generate a short document providing context for this query:\n{query}\n\nContext document:"
)

query2doc_chain = LLMChain(llm=llm, prompt=query2doc_prompt)

def query2doc(query):
    pseudo_doc = query2doc_chain.run(query)
    return pseudo_doc

# Usage
query = "What are the health benefits of turmeric?"
pseudo_doc = query2doc(query)

# 4. Multi-querying / Sub-queries
multi_query_prompt = PromptTemplate(
    input_variables=["query"],
    template="Break down this complex query into 3-5 simpler sub-queries:\n{query}\n\nSub-queries:"
)

multi_query_chain = LLMChain(llm=llm, prompt=multi_query_prompt)

def multi_query(query):
    sub_queries_text = multi_query_chain.run(query)
    sub_queries = sub_queries_text.split('\n')
    return [q.strip() for q in sub_queries if q.strip()]

# Usage
complex_query = "Compare the economic policies of FDR and Reagan"
sub_queries = multi_query(complex_query)

# Example of using these in a RAG pipeline
def rag_pipeline(query, documents):
    # Create vector store
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    docsearch = FAISS.from_documents(texts, embeddings)

    # Query rewriting
    hyde_emb = hyde(query)
    general_query = step_back(query)
    pseudo_doc = query2doc(query)
    sub_queries = multi_query(query)

    # Retrieve documents using various methods
    hyde_docs = docsearch.similarity_search_by_vector(hyde_emb, k=2)
    general_docs = docsearch.similarity_search(general_query, k=2)
    pseudo_docs = docsearch.similarity_search(pseudo_doc, k=2)
    sub_query_docs = [doc for sq in sub_queries for doc in docsearch.similarity_search(sq, k=1)]

    # Combine and deduplicate results
    all_docs = hyde_docs + general_docs + pseudo_docs + sub_query_docs
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # Here you would typically pass these documents to your LLM for final answer generation
    return unique_docs

# Usage
documents = [Document(page_content=f"Document {i}") for i in range(10)]  # Example documents
query = "What are the long-term economic impacts of climate change?"
relevant_docs = rag_pipeline(query, documents)
```

### **Understanding 'recent queries' to add temporal context**

One common application of using structured outputs for query understanding is to identify the intent of a user's query. In this example we're going to use a simple schema to seperately process the query to add additional temporal context.

```python
from datetime import date

class DateRange(BaseModel):
    start: date
    end: date

class Query(BaseModel):
    rewritten_query: str
    published_daterange: DateRange
```

In this example, `DateRange` and `Query` are Pydantic models that structure the user's query with a date range and a list of domains to search within.

These models **restructure** the user's query by including a rewritten query, a range of published dates, and a list of domains to search in.

Using the new restructured query, we can apply this pattern to our function calls to obtain results that are optimized for our backend.

```python
def expand_query(q) -> Query:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Query,
        messages=[
            {
                "role": "system",
                "content": f"You're a query understanding system for the Metafor Systems search engine. Today is {date.today()}. Here are some tips: ...",
            },
            {"role": "user", "content": f"query: {q}"},
        ],
    )

query = expand_query("What are some recent developments in AI?")
query
```

```python
Query(rewritten_query='Recent developments in artificial intelligence', published_daterange=DateRange(start=datetime.date(2024, 1, 1), end=datetime.date(2024, 3, 31)))
```

This isn't just about adding some date ranges. We can even use some chain of thought prompting to generate tailored searches that are deeply integrated with our backend.

```python
class DateRange(BaseModel):
    chain_of_thought: str = Field(
        description="Think step by step to plan what is the best time range to search in"
    )
    start: date
    end: date

class Query(BaseModel):
    rewritten_query: str = Field(
        description="Rewrite the query to make it more specific"
    )
    published_daterange: DateRange = Field(
        description="Effective date range to search in"
    )

def expand_query(q) -> Query:
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_model=Query,
        messages=[
            {
                "role": "system",
                "content": f"You're a query understanding system for the Metafor Systems search engine. Today is {date.today()}. Here are some tips: ...",
            },
            {"role": "user", "content": f"query: {q}"},
        ],
    )

expand_query("What are some recent developments in AI?")
```

```python
Query(rewritten_query='latest advancements in artificial intelligence', published_daterange=DateRange(chain_of_thought='Since the user is asking for recent developments, it would be relevant to look for articles and papers published within the last year. Therefore, setting the start date to a year before today and the end date to today will cover the most recent advancements.', start=datetime.date(2023, 3, 31), end=datetime.date(2024, 3, 31)))
```

### **Decomposing questions**

Lastly, a lightly more complex example of a problem that can be solved with structured output is decomposing questions. Where you ultimately want to decompose a question into a series of sub-questions that can be answered by a search backend. For example

"Whats the difference in populations of jason's home country and canada?"

You'd ultimately need to know a few things

1. Jason's home country
2. The population of Jason's home country
3. The population of Canada
4. The difference between the two

This would not be done correctly as a single query, nor would it be done in parallel, however there are some opportunities try to be parallel since not all of the sub-questions are dependent on each other.

```python
class Question(BaseModel):
    id: int = Field(..., description="A unique identifier for the question")
    query: str = Field(..., description="The question decomposited as much as possible")
    subquestions: List[int] = Field(
        default_factory=list,
        description="The subquestions that this question is composed of",
    )

class QueryPlan(BaseModel):
    root_question: str = Field(..., description="The root question that the user asked")
    plan: List[Question] = Field(
        ..., description="The plan to answer the root question and its subquestions"
    )

retrieval = client.chat.completions.create(
    model="gpt-4-1106-preview",
    response_model=QueryPlan,
    messages=[
        {
            "role": "system",
            "content": "You are a query understanding system capable of decomposing a question into subquestions.",
        },
        {
            "role": "user",
            "content": "What is the difference between the population of jason's home country and canada?",
        },
    ],
)

print(retrieval.model_dump_json(indent=4))
```

```python
{
    "root_question": "What is the difference between the population of Jason's home country and Canada?",
    "plan": [
        {
            "id": 1,
            "query": "What is the population of Jason's home country?",
            "subquestions": []
        },
        {
            "id": 2,
            "query": "What is the population of Canada?",
            "subquestions": []
        },
        {
            "id": 3,
            "query": "What is the difference between two population numbers?",
            "subquestions": [
                1,
                2
            ]
        }
    ]
}
```