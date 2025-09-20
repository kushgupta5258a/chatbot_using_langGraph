from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

class ChatState(TypedDict):
    message:Annotated[list[BaseMessage],add_messages]
#add_message is  a specialised reducer function 

load_dotenv()

# Initialize LLM with OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def chat_node(state:ChatState):
    #take the query from user
    message=state['message']
    #send it to llm
    response=llm.invoke(message)
    #respnse store state
    return {'message':[response]}

checkpointer=MemorySaver()
graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)