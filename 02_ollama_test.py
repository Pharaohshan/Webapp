from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

class ChatLLM:
    def __init__(self):
        self._model = ChatOllama(model="gemma2:2b", temperature=3)

        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.

        Question: {question}
        """
        self._prompt = ChatPromptTemplate.from_template(self._template)

        self._chain = (
            {'question': RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        response = self._chain.invoke(user_input)
        return response

if __name__ == "__main__":
    llm = ChatLLM()
    response = llm.invoke("인터넷 서비스의 정의는 뭐야?")
    print(response)
