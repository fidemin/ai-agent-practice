from autogen import config_list_from_json, ConversableAgent, UserProxyAgent

if __name__ == "__main__":
    config_list = config_list_from_json(
        env_or_file="./resources/autogen/config_list.json"
    )

    assistant = ConversableAgent("agent", llm_config={"config_list": config_list})
    user_proxy_agent = UserProxyAgent(
        "user",
        code_execution_config={
            "work_dir": "working",
            "use_docker": False,
        },
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
    )

    user_proxy_agent.initiate_chat(assistant, message="What is capital of France?")
