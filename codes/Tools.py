from openai import OpenAI
import os
import langchain
langchain.debug = False
from langchain.llms.base import BaseLLM
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
import configs.api_key


class ToolFunctions:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.load_info()

    def load_info(self):
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        persist_directory = os.path.join(current_dir_path, '..', 'vector_store')
        embedding = OpenAIEmbeddings()

        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding)

        self.qa = RetrievalQA.from_chain_type(
            self.llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=False
        )

    def find_vector_db(self, query: str):
        return self.qa({"query": "查询所有学生数据"})

    def get_completion_from_messages(self, messages, model="gpt-3.5-turbo", temperature=0):
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,  # gpt-4/gpt-3.5-turbo
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def create_User_Profile(self, StudentData: str):
        User_Profile_TMPL = """
        你是一名数据分析师：
        性格：专业、细心
        语言风格：正式
        任务：根据用户提供的学生数据，提取共有的特征、行为和偏好等信息，生成一句20字以内的学生画像的说明。

        注意，\
        1.学生画像的说明要针对所有学生进行总结，不能针对个别学生进行说明。\
        2.仅限于根据得到的数据生成用户画像，如果没有获得数据时：不要生成用户画像，并回复：请先获取学生数据。

        """
        messages = [
            {'role': 'user', 'content': StudentData},
            {'role': 'system', 'content': User_Profile_TMPL}]
        response = self.get_completion_from_messages(messages)
        return response

    def create_Lesson_Plan(self, profile: str):
        Lesson_Plan_TMPL = """
        你是一名幼儿绘画课程的教研老师。
        性格：专业、耐心
        语言风格：正式
        任务：根据学生画像，生成一句30字以内的面向所有学生的教学计划。\
        教学计划需说明需要提升的具体绘画能力，可以从基本形状、颜色的认知、空间的认知、颜色与情感的关联、或者阴影与立体感的绘制\
        这几项中选择2个优先级最高的能力。

        注意：\
        仅限于根据得到的画像生成教学计划，教学计划的字数不能超过30个字。\
        """
        messages = [
            {'role': 'user', 'content': profile},
            {'role': 'system', 'content': Lesson_Plan_TMPL}]
        response = self.get_completion_from_messages(messages)
        return response

    def create_Painting_Theme(self, PaintingTheme: str):
        Painting_Theme_TMPL = """
        你是名幼儿绘画课程的教研老师。根据教学计划生成1个适用所有学生的30字以内的绘画主题说明，主题说明包括主题的名称和介绍。\

        注意：
        1.绘画主题名称和介绍的内容和格式示例：\
        动物园大冒险：一个充满创意和探索的绘画主题。在这一主题中，孩子们将跟随一个勇敢的探险家进入一个神奇的动物园，\
        与各种奇特的动物互动，从中学习动物的特性、生活习性和它们所生活的环境。
        2.绘画主题名称的设计必须参考以下内容：
        ● 卡通与动画：展示不同的卡通角色或动画片段截图。\
        ● 奇特的自然景观：展示大自然中的奇观，如极光、彩虹、瀑布等，孩子们根据所看到的景观绘制。\
        ● 古代与现代：展示古代与现代的对比图片，鼓励孩子们进行对比绘画。\
        ● 世界各地的风景：展示不同国家的标志性风景图片，孩子们选择并绘制。\
        ● 建筑风格：展示来自世界各地的著名建筑图片，孩子们根据所看到的建筑风格进行绘画。\
        ● 我的超能力：让孩子们想象如果他们有超能力，那会是什么样的。\
        ● 梦中的王国：让孩子们绘制自己梦中的理想国度。\
        ● 与神秘生物的遭遇：鼓励孩子们想象与外星人或其他神秘生物的友好交往。\
        ● 我是设计师：鼓励孩子们设计未来的生活用品。\
        ● 隐藏的王国：让孩子们想象地下或海底的秘密王国。\
        ● 我和我的小伙伴：鼓励孩子们想象与动物或神话生物的冒险故事。
        """
        messages = [
            {'role': 'user', 'content': PaintingTheme},
            {'role': 'system', 'content': Painting_Theme_TMPL}]
        response = self.get_completion_from_messages(messages)
        return response

    # 图生文：生成绘画参考图
    def create_Sample_Image(self, SampleImage: str):
        CONTEXT_SampleImage_TMPL = f"""
        你是一名幼教美术老师：
        性格：活泼、创意丰富
        语言风格：亲切
        任务：生成一张面向5岁儿童，关于'{SampleImage}'主题的绘画参考样图，图片风格要积极阳光，元素要简单，适合小朋友的模仿绘画，\
        图片中不能包含任何的文字说明。
        注意：
        只能输出生成图片的地址，不能包含任何其他字符。
        """
        # print(CONTEXT_SampleImage_TMPL)

        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=CONTEXT_SampleImage_TMPL,  # 生成一张面向5岁儿童，关于‘动物园大冒险’主题相关的参考样图
            size="1024x1024",
            quality="standard",
            n=1,
        )

        # 获取图像的 URL
        image_url = response.data[0].url

        # 输出图像 URL
        # print(image_url)
        return image_url

    # 图生文：生成绘画任务话术
    def create_Task_Script(self, image_url: str):
        CONTEXT_TaskScript_TMPL = f"""
        你是一名幼教美术老师：
        性格：友善、鼓励
        语言风格：亲切
        任务：根据生成的绘画参考图的内容，生成一段不超过100个字的详细而吸引孩子们注意的任务描述。从而为孩子们解释绘画主题并鼓励绘画。\
        描述以'：亲爱的孩子们：'开头。
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CONTEXT_TaskScript_TMPL},
                        {
                            "type": "image_url",
                            "image_url": {
                                # https://img0.baidu.com/it/u=3884267628,361892966&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=348 image_url
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        # print(response.choices[0].message.content)
        TaskScript = response.choices[0].message.content
        return TaskScript

    # 图生文：生成绘画评价
    def create_Painting_Evaluation(self, image_url: str):
        CONTEXT_PaintingEvaluation_TMPL = f"""
        你是一名幼教美术老师：
        性格：公正、客观
        语言风格：正式
        任务：从绘画技术、内容和创意3个方面评价这幅由5岁小朋友创作的作品，分别给出3个方面的评分、评语，以及总分和总体评价。

        内容格式如下：

        评分（满分为10分）：

        绘画技术：5/10
        基本的面部特征已经在圆形轮廓内表达出来，虽然比例和对称性有待提高，但这是儿童绘画的典型发展阶段。
        表现了对细节如眼睛和嘴巴的关注，尽管它们的形状和位置还不完全准确。

        内容：6/10
        这幅画展示了对人脸的基础认知，孩子试图展现眼睛、鼻子和嘴巴等特征。
        能看出孩子在观察后复现人脸特征的努力，但仍需练习以更好地表达面部特征。

        创意：6/10
        尽管作品采用了简单的线条和形状，孩子展示了将观察转化为图像的创意。
        通过实验不同的形状和表达来描绘人脸，孩子表达了自己对于面部表情的理解。

        总分：17/30
        总体评价：
        这幅作品体现了5岁儿童在描绘基础人脸特征上的尝试。通过简单的线条勾勒出眼睛、鼻子和嘴巴，虽然细节的准确性还有待提高，\
        但孩子对于面部特征的观察能力和再现能力已经开始形成。这表明了他们在观察、理解和创作方面的发展潜力，是值得认可和进一步鼓励的创作实践。
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CONTEXT_PaintingEvaluation_TMPL},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        # print(response.choices[0].message.content)
        PaintingEvaluation = response.choices[0].message.content
        return PaintingEvaluation

    # 图生文：绘本故事生成
    def create_Picture_Story(self, image_url: str):
        CONTEXT_PictureStory_TMPL = f"""
        你是一名幼教美术老师：
        性格：创意丰富、温暖
        语言风格：温馨
        任务：根据这幅由5岁小朋友创作的作品，生成一个不超过200字的绘本故事。
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CONTEXT_PictureStory_TMPL},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        # print(response.choices[0].message.content)
        PictureStory = response.choices[0].message.content
        return PictureStory

    # 图生文：潜能分析
    def create_Potential_Analysis(self, image_url: str):
        CONTEXT_PotentialAnalysis_TMPL = f"""
        你是一名幼教潜能分析老师：
        性格：专业、关心
        语言风格：正式
        任务：根据这幅由5岁小朋友创作的作品，分析绘画中的情感和心理状态，发现性格潜能，生成一个不超过200字的潜能分析报告。
        """

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CONTEXT_PotentialAnalysis_TMPL},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        # print(response.choices[0].message.content)
        PotentialAnalysis = response.choices[0].message.content
        return PotentialAnalysis