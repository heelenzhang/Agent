from langchain.chat_models import ChatOpenAI
import configs.api_key
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

llm = ChatOpenAI(openai_api_key=configs.api_key.respone, temperature=0.0)

# -----------------加载PDF数据-----------------
loader = PyPDFLoader("../assets/学生数据.pdf")
pages = loader.load()

# -----------------文本分割器-----------------
# 初始化递归字符文本分割器
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\。 )", " ", ""]
)

# 分割文本
splits = r_splitter.split_documents(pages)

# 向量存储
embedding = OpenAIEmbeddings()

persist_directory = '../vector_store/'
## rm -rf 'Agents_Demo/vector_store' # 删除旧的数据库文件(如果文件夹中有文件的话)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())
print(vectordb)
