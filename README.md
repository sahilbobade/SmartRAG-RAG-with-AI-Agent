# AI Agent-Based RAG System

## Overview
This project is an AI agent-based Retrieval-Augmented Generation (RAG) system. It utilizes various tools to fetch data from vector database.

## Features
- **AI Agent Integration**: Leverages AI agents to enhance data retrieval and generation.
- **Vector Databases**: Custom-built vector databases for efficient data storage and retrieval.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/sahilbobade/RAG-with-AI-Agent.git
    ```
2. Navigate to the project directory:
    ```bash
    cd SmartRAG-RAG-with-AI-Agent
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Setting Up Vector Databases**:
    - Run the provided Python notebooks to create and populate your vector databases.
2. **Add tools in code**:
    - Edit the function create_retriever_tools in app_library.py to add tools for the new vector database that you created.
    - Alternatively you can add data to the esisting vector database using create_db_from*.py python notebooks.
3. **Running the AI Agent**:
    - run following streamlit command to run app
   ```bash
    streamlit run app.py
    ```

## Contributing
Contributions are welcome! Please contact me if you wish to contribute to the project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or suggestions, feel free to reach out:
- **Email**: sahilbobade751@gmail.com
- **GitHub**: sahilbobade

