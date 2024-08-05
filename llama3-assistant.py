import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# Defining LLM
@st.cache_resource
def load_models():
    local_llm = 'llama3:8b'
    llama3 = ChatOllama(model=local_llm, temperature=0)
    llama3_json = ChatOllama(model=local_llm, format='json', temperature=0)
    return llama3, llama3_json

llama3, llama3_json = load_models()

# Web Search Tool
@st.cache_resource
def load_search_tool():
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=50)
    return DuckDuckGoSearchRun(api_wrapper=wrapper)

web_search_tool = load_search_tool()

# Prompt Templates
@st.cache_data
def load_prompts():
    generate_prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an AI assistant for Research Question Tasks, that synthesizes web search results.
        Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know.
        Keep the answer concise, but provide all of the details you can in the form of a research report.
        Only make direct references to material if provided in the context.
        If you can't find following pieces of web search context you should answer the question as you normally would.
        
        Previous conversation:
        {history}
        
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question: {question}
        Web Search Context: {context}
        Answer:
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context", "history"],
    )
    
    router_prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert at routing a user question to either the generation stage or web search.
        Use the web search for questions that require more context for a better answer, or recent events.
        Otherwise, you can skip and go straight to the generation phase to respond.
        You do not need to be stringent with the keywords in the question related to these topics.
        Give a binary choice 'web_search' or 'generate' based on the question.
        Return the JSON with a single key 'choice' with no preamble or explanation.
        Question to route: {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    
    query_prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert at crafting web search queries for research questions.
        More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format.
        Reword their query to be the most effective web search string possible.
        Return the JSON with a single key 'query' with no preamble or explanation.
        Question to transform: {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    
    return generate_prompt, router_prompt, query_prompt

generate_prompt, router_prompt, query_prompt = load_prompts()

# Chains
@st.cache_resource
def create_chains(_llama3, _llama3_json):
    generate_chain = generate_prompt | _llama3 | StrOutputParser()
    question_router = router_prompt | _llama3_json | JsonOutputParser()
    query_chain = query_prompt | _llama3_json | JsonOutputParser()
    return generate_chain, question_router, query_chain

generate_chain, question_router, query_chain = create_chains(llama3, llama3_json)

# Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    search_query: str
    context: str
    history: str

# Node - Generate
def generate(state):
    print("Step: Generating Final Response")
    question = state["question"]
    context = state.get("context", "")
    history = state.get("history", "")
    
    if not context and not history:
        # This is the first interaction without any context
        generation = "Hello! I'm an AI research assistant. How can I help you today?"
    else:
        # Generate response using the provided context and history
        generation = generate_chain.invoke({"context": context, "question": question, "history": history})
    
    return {"generation": generation}

# Node - Query Transformation
def transform_query(state):
    print("Step: Optimizing Query for Web Search")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    search_query = gen_query["query"]
    return {"search_query": search_query}

# Node - Web Search
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_web_search(search_query):
    return web_search_tool.invoke(search_query)

def web_search(state):
    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    search_result = cached_web_search(search_query)
    return {"context": search_result}

# Conditional Edge, Routing
@st.cache_data
def cached_route_question(question):
    output = question_router.invoke({"question": question})
    return output['choice']

def route_question(state):
    print("Step: Routing Query")
    question = state['question']
    choice = cached_route_question(question)
    if choice == "web_search":
        print("Step: Routing Query to Web Search")
        return "websearch"
    elif choice == 'generate':
        print("Step: Routing Query to Generation")
        return "generate"

# Build the workflow
@st.cache_resource
def create_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.set_conditional_entry_point(route_question, {
        "websearch": "transform_query",
        "generate": "generate",
    })
    workflow.add_edge("transform_query", "websearch")
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

local_agent = create_workflow()

# Streamlit app
st.title("AI Research Assistant")

# Initialize session state for conversation history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if prompt := st.chat_input("Ask a question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare conversation history for the model
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    
    # Run the agent
    output = local_agent.invoke({
        "question": prompt,
        "history": history,
        "context": ""  # Initialize with empty context
    })
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output["generation"]})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(output["generation"])

# Display a welcome message if it's the first interaction
if not st.session_state.messages:
    st.chat_message("assistant").markdown("Hello! I'm an AI research assistant. How can I help you today?")
