import re
from typing import List, Union
import textwrap
import time
from openai import OpenAI
import langchain

langchain.debug = False # True

from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)

from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
import configs.api_key
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

from codes.Tools import ToolFunctions

def output_response(response: str) -> None:
    if not response:
        exit(0)
        # 检测文本中是否包含网址
    if "http://" in response or "https://" in response:
        print(response)
    else:
        for line in textwrap.wrap(response, width=60):
            for word in line.split():
                for char in word:
                    print(char, end="", flush=True)
                    time.sleep(0.1)
                print(" ", end="", flush=True)
            print()
    print("----------------------------------------------------------------")


AGENT_TMPL = """按照给定的格式回答以下问题。你只能使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: ”{tool_names}“ 中的其中一个工具名
Action Input: 选择工具所需要的输入
Observation: 选择工具返回的结果
...（这个思考/行动/行动输入/观察可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---

现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。\

注意：回答的内容仅限于工具返回的结果，如果返回的结果不是正确答案，不要再次尝试，也不要利用自己的知识。


Question: {input}
{agent_scratchpad}
"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。

        Returns:
            str: 填充好后的 template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")  # 取出中间步骤并进行执行
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts  # 记录下当前想法
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  # 枚举所有可使用的工具名+工具描述
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  # 枚举所有的工具名称
        cur_prompt = self.template.format(**kwargs)
        print(cur_prompt)
        return cur_prompt


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。

        Args:
            llm_output (str): _description_

        Raises:
            ValueError: _description_

        Returns:
            Union[AgentAction, AgentFinish]: _description_
        """
        if "Final Answer:" in llm_output:  # 如果句子中包含 Final Answer 则代表已经完成
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )



## set api token in terminal，gpt-4/gpt-3.5-turbo
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

Tool_Functions = ToolFunctions(llm)
tools = [
    Tool(
        name="提取学生数据",
        func=Tool_Functions.find_vector_db,
        description="当查询学生数据时，可以使用这个工具返回学生的完整数据，输入应该是需要查询的某个学生姓名或者是要查询所有学生的数据",
    ),
    Tool(
        name="生成画像",
        func=Tool_Functions.create_User_Profile,
        description=" 当需要生成画像时，可以使用这个工具返回所有学生的整体画像，输入是'提取学生数据'执行后的结果",
    ),
    Tool(
        name="生成教学计划",
        func=Tool_Functions.create_Lesson_Plan,
        description="当需要生成教学计划时，可以使用这个工具返回教学计划，输入是'生成画像'执行后的结果",
    ),
    Tool(
        name="生成绘画主题介绍",
        func=Tool_Functions.create_Painting_Theme,
        description="当需要生成绘画主题介绍时，可以使用这个工具返回绘画主题介绍，输入是'生成教学计划'执行后的结果",
    ),
    Tool(
        name="生成绘画参考图",
        func=Tool_Functions.create_Sample_Image,
        description="当需要生成绘画参考图时，可以使用这个工具返回绘画参考图，输入是'生成绘画主题介绍'执行后的结果",
    ),
    Tool(
        name="生成绘画任务话术",
        func=Tool_Functions.create_Task_Script,
        description="当需要生成绘画任务话术时，可以使用这个工具返回绘画任务话术，输入是'生成绘画参考图'执行后的结果",
    ),
    Tool(
        name="生成绘画评价",
        func=Tool_Functions.create_Painting_Evaluation,
        description="当需要生成绘画评价时，可以使用这个工具返回绘画评价，输入是用户输入的图片地址",
    ),
    Tool(
        name="生成绘本故事",
        func=Tool_Functions.create_Picture_Story,
        description="当需要生成绘本故事时，可以使用这个工具返回绘本故事，输入是用户输入的图片地址",
    ),
    Tool(
        name="生成潜能分析报告",
        func=Tool_Functions.create_Potential_Analysis,
        description="当需要生成潜能分析报告时，可以使用这个工具返回潜能分析报告，输入是用户输入的图片地址",
    ),
]
agent_prompt = CustomPromptTemplate(
    template=AGENT_TMPL,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)
output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    max_iterations=3  # 设置大模型循环最大次数，防止无限循环
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

def get_agent_executor():
    return agent_executor

def get_output_response():
    return output_response


