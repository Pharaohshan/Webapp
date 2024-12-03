import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class ChatLLM:
    def __init__(self):
        self._model = ChatOllama(model="gemma2:2b", temperature=3)

        self._template = """질문에 대해 간단하고 명확한 답변을 제공해 주세요.
        Question: {question}
        """
        self._prompt = ChatPromptTemplate.from_template(self._template)

        self._chain = (
            {"question": RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        response = self._chain.invoke(user_input)
        return response

class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def run(self):
        st.set_page_config(page_title=self._page_title, page_icon=self._page_icon)
        st.title(self._page_title)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if st.session_state["messages"]:
            for msg in st.session_state["messages"]:
                st.chat_message(msg.role).write(msg.content)

        if user_input := st.chat_input("질문을 입력해 주세요."):
            st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
            response = str(self._llm.invoke(user_input))

            st.session_state["messages"].append(ChatMessage(role="assistant", content=response))

            st.chat_message("assistant").write(response)


if __name__ == "__main__":
    llm = ChatLLM()

    web = ChatWeb(llm=llm)

    web.run()

