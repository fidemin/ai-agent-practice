import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    with open("./resources/text/ai_news_1.txt") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([raw_text])

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    client = chromadb.Client()

    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    collection = client.create_collection("news")
    collection.add(
        ids=[str(i + 1) for i in range(len(docs))],
        documents=texts,
        embeddings=vectors,
    )

    query = "Who is positive leaders for AGI?"
    vectors = embeddings.embed_documents([query])

    results = collection.query(
        query_embeddings=vectors,
        n_results=3,
    )

    for texts in results["documents"]:
        for text in texts:
            print(text)

    #
    # for i, doc in enumerate(docs):
    #     print()
    #     print(f"{i}'s doc -----------------------------------")
    #     print(doc)
