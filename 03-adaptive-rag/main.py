from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from graph.graph import app

if __name__ == "__main__":
    print("Hello Adaptive RAG Agent!")
    print(app.invoke(input={"question": "how to make pizza?"}))
