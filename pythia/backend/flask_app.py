# from langchain.agents import create_csv_agent
from langchain_experimental.agents import create_csv_agent
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import duckdb
import pandas as pd

app = Flask(__name__)

# Allow requests from 'http://localhost:3000'
CORS(app)

# Remove the max content length limit
# app.config['MAX_CONTENT_LENGTH'] = 0.2 * 1024 * 1024  # 200kb
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
sql_embeddings = OllamaEmbeddings(model="llama2")
code_embeddings = OllamaEmbeddings(model="llama3")
vectorstore_directory = 'vdb'

sql_vectorstore = Chroma(persist_directory=vectorstore_directory,
                    embedding_function=sql_embeddings)
code_vectorstore = Chroma(persist_directory=vectorstore_directory,
                    embedding_function=code_embeddings)

sql_llm = Ollama(base_url="http://127.0.0.1:11434",
             model="llama2",
             verbose=True,
             )
code_llm = Ollama(base_url="http://127.0.0.1:11434",
             model="llama3",
             verbose=True,
             )

def get_schema_information(conn):
# Get the list of tables
    tables_query = "PRAGMA show_tables;"
    tables = conn.execute(tables_query).fetchall()
    
    schema_info = []
    
    for table in tables:
        table_name = table[0]
        schema_info.append(f"Table: {table_name}")
        
        # Get the table schema
        table_info_query = f"PRAGMA table_info('{table_name}');"
        table_info = conn.execute(table_info_query).fetchall()
        
        for column in table_info:
            col_info = f"  Column: {column[1]}, Type: {column[2]}, Not Null: {column[3]}"
            schema_info.append(col_info)
    
    # Combine the schema information into a single string
    return "\n".join(schema_info)

@app.route('/chat_csv', methods=['POST'])
def chat_csv():
    sql_prompt_template = """### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Only return the sql statement in the response**
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose table name is data and the schema is represented in this string:
`{context}`
"""
    code_template = """### Instructions:
Your task is to go through the question word by word and return a response
Adhere to the following rules:
- **The response should not contain any SQL code**
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
### Input:
Generate a response that answers that answers the question `{}`.
The table name, schema and some sample data is represented in this string:
`{}`
"""
    load_dotenv()
    print("request came in")
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

    query = request.form.get('query')
    response_type = request.form.get('response_type')
    csv_file = request.files.get('csv_file')
    if csv_file is None:
        return jsonify({"error": "No CSV file provided"}), 400

    original_filename = csv_file.filename
    file_path = os.path.join(UPLOAD_FOLDER, original_filename)
    csv_file.save(file_path)

    # Read the CSV file using DuckDB
    conn = duckdb.connect(database=':memory:')
    table_nm = 'data'
    data = conn.execute(f"SELECT * FROM read_csv_auto('{file_path}')").fetchdf()
    conn.execute(f"CREATE TABLE {table_nm} AS SELECT * FROM read_csv_auto('{file_path}')").fetchdf()
    columns = ','.join(list(data.columns))
    sample_query = f'SELECT {columns} FROM data'
    print(data.columns)

    schema = get_schema_information(conn)
    sample_data = data.head().to_dict(orient='records')

    if response_type == 'Dataframe':
        SQL_PROMPT = PromptTemplate(
            template=sql_prompt_template, input_variables=["question", "context"]
        )
        sql_chain_type_kwargs = {"prompt": SQL_PROMPT}
        
    if response_type == 'Dataframe':
        sql_qa = RetrievalQA.from_chain_type(
        llm=sql_llm,
        chain_type="stuff",
        retriever=sql_vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs=sql_chain_type_kwargs 
        ,return_source_documents=True,
    )   
        
    if query is None or query == "":
        return jsonify({"error": "No user question provided"}), 400

    context = f"Table Name: {table_nm}\nSchema: {schema}\nSample Data: {sample_data}\nSample Query: {sample_query}\n"
    if response_type == 'Dataframe':
        response = sql_qa({"context": context,"query": query})
    else:
        # print(code_llm.generate([code_template.format(query, context)]).generations)
        response = code_llm.generate([code_template.format(query, context)]).generations[0][0].text

    if response_type == 'Dataframe':
        response_json = conn.execute(f"{response['result']}").fetchdf().to_json()
    else:
        response_json = response
    response_json = {"answer": response_json}
    
    return jsonify(response_json), 200
    # return result_df, 200

if __name__ == "__main__":
    app.run(debug=True)
