'''This is a dependancy for askMaintenance app.'''
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
from deep_translator import GoogleTranslator
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool



def app_config():
    '''
    This function intializes the display and UI for front end app.
    it finally returns the configuration parameters: (GPT version ,Temperature, Vebose, Voice input language, api key)
    which are selected by user on the app screen.
    '''
    # app display config
    from streamlit_mic_recorder import mic_recorder,speech_to_text
    LOGO_URL_LARGE = ""
    SMALL_LOGO = ""
    st.set_page_config(page_title="Smart Assistant", page_icon=None)
    st.header("Smart Assistant", divider="grey")
    with st.sidebar:
        # st.image(LOGO_URL_LARGE, use_column_width= True)
        st.header("Smart Assistant", divider="grey")
        api_key = st.text_input(label='Enter your OpenAI API key here', type='password')
        GPT_version = st.radio(
        "GPT version:",
        ["GPT-3.5", "GPT-4o", "GPT-4"], index =1)
        slider_temp = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
        verbose_cb = True
        speech_lang_cb = st.radio(
        "Voice input Language:",
        ["Hindi","English" ])
        if speech_lang_cb == "English":
            lang_code = 'en-IN'
        else:
            lang_code = 'hi-IN'
        st.session_state.text= speech_to_text(language=lang_code,use_container_width=True,just_once=True,key="STT")
    return GPT_version ,slider_temp, verbose_cb, lang_code, api_key

def initialize_model(GPT_version ,slider_temp, verbose_cb, api_key):
    '''
    Initialize model based on the user input.
    based on user config we select any one of available GPT version.
    It then finally returns the model object
    '''
    # Import OpenAI
    from langchain_openai.chat_models import ChatOpenAI
    #change deployement model based on user selection
    deployment_name = "gpt-35-turbo"
    if GPT_version =="GPT-3.5":
        deployment_name = "gpt-35-turbo"
    elif GPT_version == "GPT-4o":
        deployment_name = "gpt-4o"
    else:
        deployment_name = "gpt-4"

    # Create an instance of OpenAI
    model = ChatOpenAI(model= deployment_name, openai_api_key=api_key, verbose= verbose_cb, temperature=slider_temp)

    return model

def load_vectordb():
    '''
    Load vector database.

    vectordb: this database has contextual data

    It finally returns the vector database object
    '''
    #load embedding model
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    #Loading Vector database
    #chromadb 
    from langchain_chroma import Chroma
    vector_db_path = "data_db"

    manualdb = Chroma(
        persist_directory=vector_db_path, embedding_function=embeddings
    )
    return manualdb

def create_retriever_tools(vectordb):
    '''
    This function creates the retriever tools which will be used by AI agent for retrieving data.
    it return both of tools object.
    you can create multiple tools here and return as a list.
    '''
    #creating a tool
    data_retriver = vectordb.as_retriever(search_kwargs = {'k' : 2})
    data_retriever_tool = create_retriever_tool(
        data_retriver,
        name = "data_search",
        description="Search for information. For any questions, you must use this tool!",
    )
    return [data_retriever_tool]


def create_agent_prompt(tools):
    '''
    This function creates prompt needed for creating the AI agent.
    Prompt instructs AI agent about its rols.
    We pass information about tools, chat history etc. in prompt so that Agent is aware of what information and tools are at its disposal.
    
    It finally returns prompt object
    '''

    template = '''You are a Smart Assistant.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions.
    You have access to data with the help of tools. If needed retrieve required information from the vector database by using any of these tools.

    TOOLS:

    ------

    Assistant has access to the following tools:

    {tools}

    To use a tool, please use the following format:

    ```

    Thought: Do I need to use a tool? Yes

    Action: the action to take, must be one of [{tool_names}]

    Action Input: the input to the action

    Observation: the result of the action
    This cycle should not happen more that twice.

    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```

    Thought: Do I need to use a tool? No
    Don't take any action.
    Final Answer: the final answer to the original input question is the full detailed explanation from the Observation provided as bullet points.

    ```

    Begin!

    Previous conversation history:

    {chat_history}

    New input: {input}

    {agent_scratchpad}'''

    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_agent_chain(model, tools, prompt):
    '''
    This function creates the AI agent chain which will be used for answering user queries.

    1. This chain expects the user query as input
    2. It then invokes AI agent.
    3. AI agent understands the query and decides if it needs to use tools
    4. Agent goes through REACT process.
    5. Agent finally generates the answer to the user query

    This function returns the AI agent chain object.
    '''
    from langchain.chains.conversation.memory import ConversationBufferMemory

    #memory for agent
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    #instantiating agent
    from langchain.agents import AgentExecutor, create_react_agent
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors =True, memory=st.session_state.memory)


    #creating chain 
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    chain = ( {"input": RunnablePassthrough()}
            |agent_executor 
    )
    return chain

def process_input(chain, lang_code):
    '''
    This is the driver function for streamlit dynamics.
    When user gives any input (text ot voice), it invokes the AI agent chain.
    Once the resopnse is successfully generated, it is displayed on app display
    '''
    # session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am your Smart Assistant. How can I help you?"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    #handling user input
    user_query = st.chat_input("Type your message here...")
    # if there is a user text input
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            result = chain.invoke(user_query)
            response =  result["output"]
            st.write(response)

        st.session_state.chat_history.append(AIMessage(content=response))

    #if there is user speech input
    if st.session_state.text is not None:
        print(st.session_state.text)
        if lang_code == 'hi-IN':
            st.write("what you said: ",st.session_state.text)
            #using google translator for translating Hindi to english
            translation =  GoogleTranslator(source='hi', target='en').translate(st.session_state.text)
        else:
            translation = st.session_state.text
        st.session_state.chat_history.append(HumanMessage(content=translation))

        with st.chat_message("Human"):
            if lang_code == 'hi-IN':
                st.markdown("tanslated to english: " + translation)
            else:
                st.markdown( st.session_state.text)

        with st.chat_message("AI"):
            result = chain.invoke(translation )
            response = result["output"]
            st.write(response)
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.text = None


def main():
    '''Main driver function'''
    #get LLM config as per user selection
    GPT_version ,temperature, is_verbose, voice_input_language, api_key = app_config()

    #initialize model based on user config
    model = initialize_model(GPT_version ,temperature, is_verbose , api_key)

    #load vector database
    vectordb = load_vectordb()

    #create data retriever tools for AI agent
    tools = create_retriever_tools(vectordb)
    
    #create prompt for AI agent
    prompt = create_agent_prompt(tools)

    #create AI agent chain
    chain = create_agent_chain(model, tools, prompt)

    #check and process the user input
    process_input(chain, voice_input_language)

if __name__ == '__main__':
    main()



