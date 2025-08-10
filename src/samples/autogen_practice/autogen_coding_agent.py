from autogen import config_list_from_json, UserProxyAgent, AssistantAgent, Cache


def review_code(receiver, messages, sender, config):
    return f"""
    Review following code and give critics.
    
    {receiver.chat_messages_for_summary(sender)[-1]['content']}
"""


if __name__ == "__main__":
    config_list = config_list_from_json(
        env_or_file="./resources/autogen/config_list.json"
    )

    user_proxy_agent = UserProxyAgent(
        "user",
        code_execution_config={
            "work_dir": "working",
            "use_docker": False,
            "last_n_message": 1,
        },
        human_input_mode="ALWAYS",
        is_termination_msg=lambda x: x.get("content", "")
        .rstrip()
        .endswith("TERMINATE"),
    )

    engineer = AssistantAgent(
        name="Engineer",
        llm_config={
            "config_list": config_list,
        },
        system_message="""
        You are senior software engineer specialized in Python.
        You are using python to create app, tool, games.
        You prefer to write structured and simple codes to maintain them easily.
        """,
    )

    critic = AssistantAgent(
        name="Reviewer",
        llm_config={
            "config_list": config_list,
        },
        system_message="""
        You are code reviewer who is caring about standard following and strict review.
        Your mission is to review code to find something which doesn't reach the standard and secure coding.
        You give lists for improvement or suggestion of the codes.
        """,
    )

    user_proxy_agent.register_nested_chats(
        [
            {
                "recipient": critic,
                "message": review_code,
                "summary_method": "last_msg",
                "max_turns": 1,
            }
        ],
        trigger=engineer,
    )

    task = """
    create simple python script print hello
    """

    with Cache.disk(cache_seed=42) as cache:
        response = user_proxy_agent.initiate_chat(
            recipient=engineer,
            message=task,
            max_turns=2,
            summary_method="last_msg",
        )
