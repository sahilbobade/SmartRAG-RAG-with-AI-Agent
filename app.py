from app_library import *

def main():
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