import os, streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


## Set upi the Stramlit app
st.set_page_config(page_title="Rozwiązywacz zadań matematycznych i asystent wyszukiwania danych",page_icon="🧮")
st.title("Rozwiązywacz zadań matematycznych z użyciem Google Gemma 2")

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") \
               or st.sidebar.text_input("Groq API Key", type="password")
if not groq_api_key:
    st.info("Dodaj GROQ_API_KEY w Secrets lub wpisz go po lewej.")
    st.stop()


llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper(lang="pl")
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description=" Searches for information in the Polish-language Wikipedia. A tool for searching the Internet to find the vatious information on the topics mentioned."
)

## Initializa the MAth tool
math_chain=LLMMathChain.from_llm(llm=llm)

def calculator_pl(q: str) -> str:
    return math_chain.run(q + "\n\nInstructions: Respond only in Polish.")

calculator=Tool(
    name="Calculator",
    func=calculator_pl,
    description="A tools for answering math related questions. Only input mathematical expression need to be provided. Return answers in Polish."
)

prompt="""
Your a agent tasked for solving users mathemtical question. Think step by step and provide a solution with bulleted justification. Always respond ONLY in Polish.
Question:{question}
Answer:
""".strip()

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

# def reasoning_pl(q: str) -> str:
#     return chain.run(q)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for gives full, point-by-point explanations in Polish."
)

# ## Agent z polskim systemowym wstępem (prefix)
# SYSTEM_PREFIX = (
#     "You are a helpful assistant. First, try using the tool"
#     "'Reasoning Tool' to prepare a complete, step-by-step explanation. "
#     "You may use the 'Calculator' for auxiliary calculations. "
#     "The final answer MUST be in Polish and include bulleted steps and a clearly stated number at the end as the result."
# )

## initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # agent_kwargs={"prefix": SYSTEM_PREFIX},
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Cześć, jestem chatbotem matematycznym, który odpowie na wszystkie Twoje pytania z matematyki"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## LEts start the interaction
question=st.text_area("Enter youe question:","")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
                                         )
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Proszę wpisać pytanie")









