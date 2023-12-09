from src.Agent import get_agent_executor, get_output_response

def run():
    agent_executor = get_agent_executor()
    output_response = get_output_response()

    while True:
        try:
            user_input = input("有什么可以帮助您的吗？你可以说：\n1.帮我生成一个绘画参考图和任务话术。\n2.帮我评价一下这幅绘画作品，然后生成一段绘本故事和潜能报告。"
                               "图片地址：https://github.com/heelenzhang/Agent/blob/main/assets/demo_pic_2.jpeg?raw=true\n请输入您的问题：")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run()