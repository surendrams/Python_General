from openai import OpenAI
import json
from rich import print as rprint

# === CONFIGURATION ===
client = OpenAI()

DB_CONFIG = {
    "host": "your_host",
    "port": "1521",
    "service_name": "your_service_name",
    "user": "your_username",
    "password": "your_password"
}

SCHEMA_FILE = {
  "tables": {
    "orders": {
      "columns": {
        "order_id": "NUMBER",
        "customer_id": "NUMBER",
        "order_date": "DATE",
        "total_amount": "NUMBER"
      },
      "relationships": {
        "customer_id": {
          "references": "customers.customer_id"
        }
      }
    },
    "customers": {
      "columns": {
        "customer_id": "NUMBER",
        "name": "VARCHAR2",
        "email": "VARCHAR2"
      }
    }
  }
}

# === LOAD SCHEMA INFORMATION ===
def load_schema():
    return SCHEMA_FILE

# === GENERATE SQL USING OPENAI ===
def generate_sql(user_query, schema):
    # Convert schema dictionary to a readable format for prompt
    schema_prompt = json.dumps(schema, indent=2)
    
    prompt = (
        f"Schema:\n{schema_prompt}\n\n"
        f"User Query: \"{user_query}\"\n"
        "Convert this to an optimized SQL query and show only the query:"
    )
    
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "text"
            },
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    #print(f'{response}')
    if response and response.choices:
        sql_query = response.choices[0].message.content.strip('`').replace('sql', '')  # Ensure clean output
        print(f'|{sql_query}|')
        return sql_query
    else:
        return "Not sure about your question"

# === EXECUTE SQL ON ORACLE DB ===
def execute_sql(sql_query):
    # dsn = cx_Oracle.makedsn(DB_CONFIG["host"], DB_CONFIG["port"], service_name=DB_CONFIG["service_name"])
    # connection = cx_Oracle.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=dsn)
    # cursor = connection.cursor()
    
    print(f"Executing SQL: {sql_query}")
    # cursor.execute(sql_query)
    # results = cursor.fetchall()
    
    # cursor.close()
    # connection.close()
    
    return 0 #results

# === MAIN FUNCTION ===
def main():
    schema = load_schema()
    user_query = input("Ask your question: ")
    
    sql_query = generate_sql(user_query, schema)
    rprint("\nGenerated SQL Query:\n", sql_query)
    
    #results = execute_sql(sql_query)
    #print("\nQuery Results:")
    #for row in results:
    #    print(row)

if __name__ == "__main__":
    main()

