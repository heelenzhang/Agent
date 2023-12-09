from codes.Agent import get_agent_executor, get_output_response

def run():
    agent_executor = get_agent_executor()
    output_response = get_output_response()

    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent_executor.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run()